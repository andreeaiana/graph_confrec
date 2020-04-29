# graph_confrec

The source code for the paper "GraphConfRec: A Graph-Neural-Network-Based Conference Recommender System"

## Overview

This directory contains the code necessary to run GraphConfRec. We provide the implementation of various recommendation models, along with evaluation scripts. 
The repository is organised as follows:
 - `data/`: contains the necessary raw dataset files for GraphConfRec and stores the processed data, including trained models
 - `notebooks/`: contains Jupyter Notebooks for exporing the SciGraph, WikiCfP, and Google H5 Index datasets
 - `src/`: contains the implementation of the recommendation models, evaluation and data preprocessing scripts

Further instructions on obtaining and preprocessing the raw data, as well as on running the code, can be found in the respective folders.

## Recommendation Models

We implemented the following recommendation models:

| **Base Model** | **Recommendation model** | 
| :-------------------------: | :-------------------------------------------: | 
| Authors                     |  Authors   |  
| Unsupervised GraphSAGE      |  GraphSAGE Neighbour, GraphSAGE Classifier (citations graph), GraphSAGE Classifier (co-authorship graph), GraphSAGE Concat  |     
| Supervised GraphSAGE        |  GraphSAGE supervised (citations graph),  GraphSAGE supervised (heterogeneous graph)   |  
| Unsupervised GraphSAGE_RL   |  GraphSAGE_RL Classifier (citations graph)    |   
| Supervised GraphSAGE_RL     |  GraphSAGE_RL supervised (citations graph),  GraphSAGE_RL supervised (heterogeneous graph)    |  
| ASGCN   					  |  ASGCN     | 
| GAT     					  |  GAT (citations graph), GAT (heterogeneous graph) |
| HAN   					  |  HAN     | 
| SciBERT + ARGA    		  |  SciBERT + ARGA (citations graph), SciBERT + ARGA (heterogeneous graph) |
  

## Dependencies

The code was tested running under Python 3.6.8, with the following packages installed (along with their dependencies):
 
 - `numpy==1.16.2`
 - `pandas==0.24.1`
 - `scipy==1.2.1`
 - `networkx==2.2`
 - `beautifulsoup4==4.7.1`
 - `tensorflow==2.0.0`
 - `tensorflow-gpu==2.0.0`
 - `torch==1.3.0`
 - `torch-cluster==1.4.5`
 - `torch-geometric==1.3.2`
 - `torch-scatter==1.4.0`
 - `torch-sparse==0.4.3`
 - `transformers==2.1.1`
 - `Flask==1.1.1`
 - `Flask-SQLAlchemy==2.4.1`
 - `SQLAlchemy==1.3.1`
 
In addition, CUDA 10.0 was used.


## License
The MIT license is applied to the provided source code.
For the datasets, please check the licensing information:
 - See https://scigraph.springernature.com/explorer/license/ for SciGraph.
 - See http://www.wikicfp.com/cfp/data.jsp for WikiCfP.

Parts of the code were originally forked from:
 - [GraphSAGE](https://github.com/williamleif/GraphSAGE/) 
 - [Advancing GraphSAGE with A Data-driven Node Sampling](https://github.com/oj9040/GraphSAGE_RL)  
 - [AS-GCN in Tensorflow](https://github.com/huangwb/AS-GCN/) 
 - [Graph Attention Networks](https://github.com/PetarV-/GAT)
 - [Heterogeneous Graph Neural Network](https://github.com/Jhy1993/HAN) 

We owe many thanks to the authors of the different models for making their codes available.
