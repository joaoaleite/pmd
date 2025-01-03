import os
import requests
import pandas as pd
import torch
import numpy as np
from bs4 import BeautifulSoup
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from dotenv import load_dotenv
from utils import SemEvalClassifier, SEMEVAL_LABELS

load_dotenv()

# Environment variables
cuda_order = os.getenv("CUDA_DEVICE_ORDER")
cuda_device = os.getenv("CUDA_DEVICE_NUM")
team = os.getenv("TEAM")
passcode = os.getenv("PASSCODE")
api_url = os.getenv("API_URL")

os.environ["CUDA_DEVICE_ORDER"], os.environ["CUDA_VISIBLE_DEVICES"] = (
    cuda_order,
    cuda_device,
)

# Select the checkpoint with highest score
model_checkpoints = os.listdir("classifier/checkpoints/")
model_scores = [
    (i, float(checkpoint.split("_")[1].split(".ckpt")[0]), checkpoint.split("_")[0])
    for i, checkpoint in enumerate(model_checkpoints)
]
MODEL_PATH = f"classifier/checkpoints/{model_checkpoints[max(model_scores, key=lambda x: x[1])[0]]}"
PRETRAINED_NAME = model_scores[max(model_scores, key=lambda x: x[1])[0]][2]

multibin = MultiLabelBinarizer().fit([SEMEVAL_LABELS])


# Submit predictions
def submit_results(lang: str, results: str):
    if lang not in ["en"]:
        raise ValueError("Invalid language")

    subtask = "3"
    resp = requests.post(
        api_url,
        data={
            "team": team,
            "passcode": passcode,
            "dataset": "dev",
            "task": f"{lang}{subtask}",
        },
        files={"sub": (f"{subtask}-{lang}.txt", results)},
        headers={"User-Agent": "Mozilla/5.0"},
    )
    bs = BeautifulSoup(resp.text, "html.parser")
    body_text = bs.body.get_text("\n")

    if "Prediction file format is correct" not in body_text:
        print("Error from submission server:\n", body_text, "\n=================")
        return None

    _, f1_micro, f1_macro = tuple(
        [td.text for td in bs.find("table").find_all("tr")[1].find_all("td")]
    )

    return f1_micro, f1_macro


def submit_from_folder(df):
    results = df.to_csv(index=True, header=False, sep="\t")
    micros, macros = zip(*[submit_results(lang, results) for lang in ["en"]])
    return pd.DataFrame({"lang": ["en"], "f1-micro": micros, "f1-macro": macros})


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SemEvalClassifier.load_from_checkpoint(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME)
    print(f"Loaded model from checkpoint '{MODEL_PATH}'")

    df = pd.read_json("datasets/raw/semeval2023_persuasion_dataset.json", lines=True)
    test_df = df[(df["lang"] == "en") & (df["split"] == "test")].set_index(
        ["id", "line"]
    )

    # Tokenize and prepare batches
    test_tokens = tokenizer(
        test_df["text"].tolist(),
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    data_loader = DataLoader(
        TensorDataset(test_tokens["input_ids"], test_tokens["attention_mask"]),
        batch_size=8,
    )

    # Run inference
    predictions = [
        (
            torch.sigmoid(
                model(
                    input_ids=batch[0].to(device), attention_mask=batch[1].to(device)
                ).logits
            )
            >= 0.5
        )
        .cpu()
        .numpy()
        for batch in data_loader
    ]
    predictions = np.vstack(predictions).astype(int)

    # Transform and submit predictions
    formatted_preds = pd.DataFrame(
        [",".join(labels) for labels in multibin.inverse_transform(predictions)],
        index=test_df.index,
    )
    print(submit_from_folder(formatted_preds))


if __name__ == "__main__":
    run_inference()
