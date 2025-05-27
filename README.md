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
```
@inproceedings{leite-2025-cross-domain-persuasion,
author = {Leite, Jo\~{a}o A. and Razuvayevskaya, Olesya and Scarton, Carolina and Bontcheva, Kalina},
title = {A Cross-Domain Study of the Use of Persuasion Techniques in Online Disinformation},
year = {2025},
isbn = {9798400713316},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3701716.3715535},
doi = {10.1145/3701716.3715535},
abstract = {Disinformation, irrespective of domain or language, aims to deceive or manipulate public opinion, typically employing advanced persuasion techniques. Qualitative and quantitative research on the weaponisation of persuasion techniques in disinformation narratives, however, has been mostly limited to specific topics (e.g., COVID-19).To address this gap, our study conducts a large-scale, multi-domain analysis of the role of 16 persuasion techniques in disinformation narratives, by leveraging a state-of-the-art persuasion technique classifier. We demonstrate how different persuasion techniques are employed disproportionately in different disinformation domains. We also include an in-depth case study on climate change disinformation, which demonstrates how linguistic, psychological, and cultural factors shape the adaptation of persuasion strategies to fit unique thematic contexts.},
booktitle = {Companion Proceedings of the ACM on Web Conference 2025},
pages = {1100–1103},
numpages = {4},
keywords = {disinformation, domain adaptation, persuasion techniques},
location = {Sydney NSW, Australia},
```
series = {WWW '25}
}
