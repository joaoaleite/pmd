# A Cross-Domain Study of the Use of Persuasion Techniques in Online Disinformation

This repository contains the experiments and supplementary material to our research paper published at The Web Conf (WWW) 2025.

## Install dependencies
```bash
conda create -n pmd python=3.9
conda activate pmd
pip install -r requirements.txt  
```

## Reproducing the analysis
1. Download the four datasets (CIDII, EUvsDisinfo, Climate Fever, and CoAID). Save them in a folder named 'datasets/raw/\<DATASET>.csv', where DATASET is {euvsdisinfo, coaid, climate_fever, cidii}. Then, preprocess the data by running: 
    > python3 normalise_datasets.py
2. Infer the persuasion techniques. <br>
   a. Go to the [GATE Cloud Persuasion API](https://cloud.gate.ac.uk/shopfront/displayItem/persuasion-classifier). Create an account, login, and create API keys. Save your api keys in a file named '.env' in the root directory of this repository. The .env file should contain the variable GATE_API_KEY="<YOUR_KEY>". <br>
   b. Run the inference script calling the Persuasion API:

   > python3 infer_api.py

3. Run the analysis notebook in `eda.ipynb`. You must use the [LIWC-22 software](https://www.liwc.app/) to extract the LIWC features in order to reproduce the contextual analysis of persuasion techniques.

## Cite
TBA
