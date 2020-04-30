# Data

## SciGraph 

To parse the raw SciGraph data:

```python

import FileParser

parser = FileParser.FileParser()
parser.get_data(process = $DATASET) # parses data as specified by process

```
| **Parameter** | **Description** | **Options** | 
| :-----------: | :------------- | :--------- |
| process | The datasets to be parsed | old_books, old_books_new_books, old_books_conferences, conferences, conferences_name, conferences_acronym, conferences_city, conferences_country, conferences_year, conferences_datestart, conferences_dateend, conferences_conferenceseries, conferenceseries, conferenceseries_name, books, isbn_books, authors_name, chapters, chapters_title, chapters_year, chapters_language, chapters_abstract, chapters_authors, chapters_authors_name, chapters_all_citations, chapters_keywords, chapters_books_isbns |

To map files from the two SciGraph releases and further process the parsed raw data:

```python

import DatasetsParser

parser = DatasetsParser.DatasetsParser()
parser.get_data(process = $DATASETS) # parses data as specified by process

```

| **Parameter** | **Description** | **Options** | 
| :-----------: | :------------- | :--------- |
| process | The datasets to be parsed | chapters_books, chapters_all_scigraph_citations, chapters_confproc_scigraph_citations, books_conferences, author_id_chapters,  author_name_chapters, confproc_scigraph_citations_chapters|



To load parsed data:

```python

import DataLoader

loader = DataLoader.DataLoader()
loader.papers(years="2016") # loads all conference proceedings from 2016
loader.papers(years="2016").conferences().conferenceseries() # loads all conference proceedings, with corresponding conferences and conference series from 2016
loader.training_data_with_abstracts_citations() # loads training data, including abstracts and citations
```



## WikiCfP 

1. Crawling the WikiCfP dataset: `python WikiCFPCrawler.py start_eventid $START_EVENTID end_eventid $END_EVENTID`
2. Linking the dataset to SciGraph: `python WikiCFPLinker.py --similarity_metric $SIMILARITY_METRIC --match_threshold $MATCH_THRESHOLD --remove_stopwords $REMOVE_STOPWORDS`
3. Evaluating the linking: `python WikiCFPLinkerEvaluation.py --similarity_metric $SIMILARITY_METRIC --match_threshold $MATCH_THRESHOLD --remove_stopwords $REMOVE_STOPWORDS`


| **Parameter** | **Description** | **Default** | **Options** | **Mandatory** |
| :-----------: | :------------- | :---------: | :--------- | :-----------: |
| start_eventid | The event ID from which to start crawling. | - | integer value | Yes |
| end_eventid | he event ID at which to stop crawling | - | integer value | Yes |
| similarity_metric | The similarity metric used | damerau_levenshtein |  levenshtein, damerau_levenshtein, jaro, jaro_winkler | No |
| match_threshold | The similarity threshold to matching two entities | 0.9 | continuous value in (0,1) | No |
| remove_stopwords | Whether to remove stopwords | True | boolean | No|


## H5Index

1. Crawling the H5Index dataset from Google Scholar Metrics: `python H5IndexScraper.py`
2. Linking the dataset to SciGraph: `python H5IndexLinker.py --similarity_metric $SIMILARITY_METRIC --threshold $THRESHOLD`
3. Evaluating the linking: `python H5IndexLinkerEvaluation.py --similarity_metric $SIMILARITY_METRIC --threshold $THRESHOLD`

| **Parameter** | **Description** |**Default** | **Options** | **Mandatory** |
| :-----------: | :------------- | :--------: | :--------- | :-----------: |
| similarity_metric | The similarity metric used | damerau_levenshtein |  levenshtein, damerau_levenshtein, jaro, jaro_winkler | No |
| threshold | 0.9 | The similarity threshold to matching two entities | continuous value in (0,1) | No |
