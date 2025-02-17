import pandas as pd
from classify_api import PersuasionClassifier
import os


def format_predictions(api_results):
    """
    Convert API results to the same format as the original infer.py
    (comma-separated persuasion techniques)
    """
    techniques = []
    for result in api_results:
        result_techniques = list(result.get("entities", {}).keys())
        techniques.append(",".join(result_techniques) if result_techniques else "")
    return techniques


def run_inference(df, batch_size=32):
    """
    Run inference on a dataframe using the GATE Cloud API
    """
    classifier = PersuasionClassifier()

    print(f"Processing {len(df)} texts...")
    results = classifier.classify_batch(df["text"].tolist(), batch_size=batch_size)

    df["persuasion_techniques"] = format_predictions(results)
    return df


if __name__ == "__main__":
    print("Loading datasets...")
    datasets = {
        "cidii": pd.read_csv("datasets/normalised/cidii.csv"),
        "coaid": pd.read_csv("datasets/normalised/coaid.csv"),
        "climate_fever": pd.read_csv("datasets/normalised/climate_fever.csv"),
        "euvsdisinfo": pd.read_csv("datasets/normalised/euvsdisinfo.csv"),
    }

    os.makedirs("datasets/processed", exist_ok=True)
    for name, df in datasets.items():
        print(f"\nProcessing {name} dataset...")
        processed_df = run_inference(df)

        output_path = f"datasets/processed/api_{name}.csv"
        processed_df.to_csv(output_path, index=False)
        print(f"Saved processed {name} dataset to {output_path}")
