# Overview

This directory is organised as follows:
 - `data/`: contains scripts for preprocessing the data, crawling the WikiCfP and H5Index datasets and linking them to the SciGraph datasets
 - `models/`: contains the implementation of the recommendation models and evaluation scripts
	- `AbstractClasses.py`: abstract classes to supervise models' querying and evaluation
	- `evaluation/`: contains implementation of evaluation metrics
	-  _one folder per model_ with:
		- classes implementing different parts of the base GNN models
		- script for training the base GNN model
		- `<model_name>Model.py`: implements the _query_single_ and _query_batch_ methods
		- `<model_name>ModelEvaluation.py`: evaluates the recommendation model on the test set
 - `user_interface/`: contains the implementation of the user interface and corresponding data
 - `utils/`: contains `TimerCounter.py`, implements a timer to be used when training the models
 
# Running the code

1. [Data](./data/README.md)
2. [Models](./models/README.md)
3. [User Interface](./user_interface/README.md)

