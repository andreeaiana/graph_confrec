# Data

This directory is organised as follows:
 - `external/`: contains the pre-trained SciBERT language model
 - `interim/`: contains intermediary data files
	- `parsed_data/`: folder for the parsed SciGraph raw dataset files
	- `WikiCfP/`: folder for the crawled dataset from WikiCfP, corresponding gold standard, and matched conference series from SciGraph
	- `H5Index/`: folder for the crawled dataset from Google Scholar Metrics, corresponding gold standard, and matched conference series from SciGraph 
	- `gat/`: folder for the training graphs and files for the GAT and ASGCN-based recommendation models
	- `graphsage/`: folder for the training graphs and files for the GraphSAGE and GraphSAGE_RL-based recommendation models
	- `han/`: folder for the training graphs and files for the HAN-based recommendation models
	- `scibert_arga/`: folder for the training graph and files for the ARGA models, and the pre-trained SciBERT embeddings of the abstracts and ARGA embeddings of the graph nodes for the SciBERT + ARGA-based recommendation models
 - `processed/`: contains the trained recommendation models
	- `as_gcn/`: folder for trained ASGCN-based recommendation models
	- `gat/`: folder for trained GAT-based recommendation models
	- `graphsage/`: folder for trained GraphSAGE-based recommendation models
	- `graphsage_rl/`: folder for trained GraphSAGE_RL-based recommendation models
	- `han/`: folder for trained HAN-based recommendation models
	- `scibert_arga/`:	folder for trained SciBERT + ARGA-based recommendation models
 - `raw/`: contains the raw SciGraph datasets
 
### Downloading the raw datasets

The _books, _chapters, and _persons dataset files from the **2019 Q1 SciGraph release** can be downloaded from https://sn-scigraph.figshare.com/ .

The _books and _conferences dataset files **2018 Q1 SciGraph release** can be downloaded from https://drive.google.com/drive/folders/10gH12snkWsbPQaE3R4P6CMFTzaxzHtqq?usp=sharing .

### Pre-trained SciBERT model

In our experiments, we used the **scibert-scivocab-uncased** model for PyTorch HuggingFace.
To replicate the experiments with the same pre-trained SciBERT model, download it from https://github.com/allenai/scibert and unzip it in `external/scibert_scivocab_uncased`.