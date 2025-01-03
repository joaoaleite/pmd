import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from dotenv import load_dotenv
from utils import SemEvalClassifier, SEMEVAL_LABELS
from tqdm import tqdm

load_dotenv()

# Environment variables
cuda_order = os.getenv("CUDA_DEVICE_ORDER")
cuda_device = os.getenv("CUDA_DEVICE_NUM")

os.environ["CUDA_DEVICE_ORDER"], os.environ["CUDA_VISIBLE_DEVICES"] = (
    cuda_order,
    cuda_device,
)

print("Using CUDA DEVICE:", os.environ["CUDA_VISIBLE_DEVICES"])
# Select the checkpoint with highest score
model_checkpoints = os.listdir("classifier/checkpoints/")
model_scores = [
    (i, float(checkpoint.split("_")[1].split(".ckpt")[0]), checkpoint.split("_")[0])
    for i, checkpoint in enumerate(model_checkpoints)
]
MODEL_PATH = f"classifier/checkpoints/{model_checkpoints[max(model_scores, key=lambda x: x[1])[0]]}"
PRETRAINED_NAME = model_scores[max(model_scores, key=lambda x: x[1])[0]][2]

multibin = MultiLabelBinarizer().fit([SEMEVAL_LABELS])


def run_inference(df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SemEvalClassifier.load_from_checkpoint(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME)
    print(f"Loaded model from checkpoint '{MODEL_PATH}'")

    # Tokenize and prepare batches
    tokens = tokenizer(
        df["text"].tolist(),
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    data_loader = DataLoader(
        TensorDataset(tokens["input_ids"], tokens["attention_mask"]), batch_size=8
    )

    # Run inference
    predictions = []
    for batch in tqdm(data_loader, total=len(data_loader)):
        out = model(
            input_ids=batch[0].to(device), attention_mask=batch[1].to(device)
        ).logits
        out = torch.sigmoid(out)
        out = (out >= 0.2).cpu().numpy()
        predictions.append(out)

    # predictions = [
    #     (torch.sigmoid(model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device)).logits) >= 0.2)
    #     .cpu()
    #     .numpy()
    #     for batch in data_loader
    # ]
    predictions = np.vstack(predictions).astype(int)

    formatted_preds = pd.DataFrame(
        [",".join(labels) for labels in multibin.inverse_transform(predictions)],
        index=df.index,
    )

    df["persuasion_techniques"] = formatted_preds
    return df


if __name__ == "__main__":
    cidii_df = pd.read_csv("datasets/normalised/cidii.csv")
    coaid_df = pd.read_csv("datasets/normalised/coaid.csv")
    climate_fever_df = pd.read_csv("datasets/normalised/climate_fever.csv")
    covid_df = pd.read_csv("datasets/normalised/covid.csv")
    euvsdisinfo_df = pd.read_csv("datasets/normalised/euvsdisinfo.csv")

    cidii_df = run_inference(cidii_df)
    coaid_df = run_inference(coaid_df)
    climate_fever_df = run_inference(climate_fever_df)
    covid_df = run_inference(covid_df)
    euvsdisinfo_df = run_inference(euvsdisinfo_df)

    cidii_df.to_csv("datasets/processed/cidii.csv", index=False)
    coaid_df.to_csv("datasets/processed/coaid.csv", index=False)
    climate_fever_df.to_csv("datasets/processed/climate_fever.csv", index=False)
    covid_df.to_csv("datasets/processed/covid.csv", index=False)
    euvsdisinfo_df.to_csv("datasets/processed/euvsdisinfo.csv", index=False)
