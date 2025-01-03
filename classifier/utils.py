# train_utils.py

import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd

SEMEVAL_LABELS = [
    "Appeal_to_Authority",
    "Appeal_to_Fear-Prejudice",
    "Appeal_to_Hypocrisy",
    "Appeal_to_Popularity",
    "Appeal_to_Time",
    "Appeal_to_Values",
    "Causal_Oversimplification",
    "Consequential_Oversimplification",
    "Conversation_Killer",
    "Doubt",
    "Exaggeration-Minimisation",
    "False_Dilemma-No_Choice",
    "Flag_Waving",
    "Guilt_by_Association",
    "Loaded_Language",
    "Name_Calling-Labeling",
    "Obfuscation-Vagueness-Confusion",
    "Questioning_the_Reputation",
    "Red_Herring",
    "Repetition",
    "Slogans",
    "Straw_Man",
    "Whataboutism",
]

# Label mapping
id2label = {i: label for i, label in enumerate(SEMEVAL_LABELS)}
label2id = {label: i for i, label in enumerate(SEMEVAL_LABELS)}


def load_train_dev_semeval():
    df = pd.read_json(
        "datasets/semeval_translated_en.json", lines=True, orient="records"
    )
    df_train = df[df["split"] == "train"]
    df_dev = df[df["split"] == "dev"]

    return df_train, df_dev


def get_dataloader(data, tokenizer, batch_size):
    encodings = tokenizer(
        data["text"].tolist(), truncation=True, padding=True, max_length=512
    )
    dataset = SemEvalDataset(encodings, data["labels"].tolist())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


# Dataset class
class SemEvalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.labels)


# Lightning module
class SemEvalClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate, clf_threshold):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.threshold = clf_threshold

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def compute_metrics(self, logits, labels):
        preds = (torch.sigmoid(logits).float() > self.threshold).int()
        labels = labels.int()

        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()

        f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        precision, recall, _, _ = precision_recall_fscore_support(
            labels, preds, average="micro"
        )

        return f1_micro, f1_macro, precision, recall

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits
        labels = batch["labels"]
        f1_micro, f1_macro, precision, recall = self.compute_metrics(logits, labels)

        self.log(
            "eval_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "eval_f1_micro",
            f1_micro,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "eval_f1_macro",
            f1_macro,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "eval_precision",
            precision,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "eval_recall",
            recall,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
