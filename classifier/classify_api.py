import requests
import json
from typing import List, Dict, Union
import os
from dotenv import load_dotenv


class PersuasionClassifier:
    """Class to handle persuasion classification using the GATE Cloud API"""

    def __init__(self):
        self.api_url = "https://cloud-api.gate.ac.uk/process/persuasion-classifier"
        load_dotenv()
        self.api_key = os.getenv("GATE_API_KEY")
        if not self.api_key:
            raise ValueError("GATE_API_KEY environment variable not found")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def classify_text(self, text: str) -> Dict[str, Union[str, List[Dict]]]:
        """
        Classify a single text using the GATE Cloud API.

        Args:
            text (str): The text to classify

        Returns:
            dict: The API response containing classification results
        """
        try:
            payload = {"text": text, "threshold": 0.2}
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error making API request: {e}")
            return None

    def classify_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Classify a batch of texts using the GATE Cloud API.

        Args:
            texts (List[str]): List of texts to classify
            batch_size (int): Number of texts to process in each batch

        Returns:
            List[Dict]: List of classification results for each text
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = []

            for text in batch:
                result = self.classify_text(text)
                if result:
                    batch_results.append(result)

            results.extend(batch_results)
            print(
                f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
            )

        return results


def main():
    classifier = PersuasionClassifier()

    test_text = (
        "You should really consider buying this product. It's the best in the market!"
    )
    result = classifier.classify_text(test_text)
    print("Single text classification result:")
    print(json.dumps(result, indent=2))

    test_texts = [
        "You should really consider buying this product.",
        "This is the best decision you'll ever make!",
        "Studies show that this approach works better.",
    ]
    results = classifier.classify_batch(test_texts)
    print("\nBatch classification results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
