# Data

## SciGraph 


## WikiCfP 


## H5Index

1. Crawling the H5Index dataset from Google Scholar Metrics: `python H5IndexScraper.py`
2. Linking the dataset to SciGraph: `python H5IndexLinker.py --similarity_metric similarity_metric --threshold threshold`
3. Evaluating the linking: `python H5IndexLinkerEvaluation.py --similarity_metric similarity_metric --threshold threshold`

| **Parameter** | **Deafult** | **Options** | **Mandatory** |
| :-----------: | :---------: | :---------: | :-----------: |
| similarity_metric | damerau_levenshtein |  levenshtein, damerau_levenshtein, jaro, jaro_winkler |  |
| threshold | 0.9 | continuous value in (0,1) | |
