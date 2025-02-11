# %%
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import re


def remove_urls(text):
    url_pattern = r"https?://\S+|www\.\S+"

    cleaned_text = re.sub(url_pattern, "", text)
    return cleaned_text


def clean(text):
    def remove_urls(text):
        url_pattern = r"https?://\S+|www\.\S+"

        cleaned_text = re.sub(url_pattern, "", text)
        return cleaned_text

    text = remove_urls(text)

    # remove very small texts
    if len(text) < 20:
        return False

    # remove texts with only numbers
    if text.isdigit():
        return False

    return True


# %%
nltk.download("punkt")
nltk.download("punkt_tab")

# %%
df = pd.read_json(
    "datasets/raw/climate-fever-dataset-r1.jsonl", lines=True, orient="records"
)
df = df[(df["claim_label"] == "REFUTES") | (df["claim_label"] == "SUPPORTS")]

df["claim_label"] = df["claim_label"].map({"REFUTES": 1, "SUPPORTS": 0})
df["text"] = df["claim"].apply(lambda x: sent_tokenize(x))
df = df.explode("text").reset_index()
df["sentence_index"] = df.groupby("index").cumcount()
df = df.rename(
    {"index": "id_article", "sentence_index": "id_sentence", "claim_label": "label"},
    axis=1,
)
df = df[["id_article", "id_sentence", "text", "label"]]
df = df[df["text"].apply(clean)].reset_index(drop=True)
df.to_csv("datasets/normalised/climate_fever.csv", index=False)

# %%
df = pd.read_csv("datasets/raw/cidii.csv")
df["sentences"] = df["Article"].apply(sent_tokenize)

df = df.explode("sentences").reset_index()
df["sentence_index"] = df.groupby("index").cumcount()
df.rename(
    {
        "sentences": "text",
        "sentence_index": "id_sentence",
        "ID": "id_article",
        "Disinformation": "label",
    },
    axis=1,
    inplace=True,
)
df = df[["id_article", "id_sentence", "text", "label"]]
df = df[df["text"].apply(clean)].reset_index(drop=True)
df.to_csv("datasets/normalised/cidii.csv", index=False)

paths = [
    f"datasets/raw/Constraint_English_{split}.csv" for split in ["Train", "Test", "Val"]
]
dfs = [pd.read_csv(path) for path in paths]
for df in dfs:
    df["label"] = df["label"].map({"real": 0, "fake": 1})
df = pd.concat(dfs, ignore_index=True)
df["text"] = df["tweet"].apply(lambda x: sent_tokenize(x))
df = df.explode("text").reset_index()
df["sentence_index"] = df.groupby("index").cumcount()
df = df[["index", "sentence_index", "text", "label"]]
df = df.rename({"index": "id_article", "sentence_index": "id_sentence"}, axis=1)
df = df[df["text"].apply(clean)].reset_index(drop=True)
df.to_csv("datasets/normalised/covid.csv", index=False)

# %%
df = pd.read_csv("datasets/raw/euvsdisinfo_translated.csv")
df["text"] = df["text"].apply(lambda x: sent_tokenize(x))
df = df.explode("text").reset_index()
df["sentence_index"] = df.groupby("index").cumcount()
df = df[
    [
        "article_id",
        "published_date",
        "keywords",
        "sentence_index",
        "text",
        "class",
    ]
]
df = df.rename(
    {"article_id": "id_article", "sentence_index": "id_sentence", "class": "label"},
    axis=1,
)
df = df[df["text"].apply(clean)].reset_index(drop=True)
df["label"] = df["label"].apply(lambda x: 1 if x == "disinformation" else 0)
df.to_csv("datasets/normalised/euvsdisinfo_translated.csv", index=False)
