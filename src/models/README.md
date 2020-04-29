# Models

## Create training data

For _ASGCN_-based recommendation models, run:

```
python preprocess_data.py embedding_type $EMBEDDING_TYPE dataset $DATASET --gpu $GPU
```

| **Parameter** | **Description** | **Default** | **Options** | 
| :-----------: | :------------- | :----------: | :---------- |
| embedding_type | Type of SciBERT embedding | - | AVG_L, AVG_2L, AVG_SUM_ALL, AVG_SUM_L4, CONC_AVG_MAX_2L, CONC_AVG_MAX_SUM_L4, MAX_2L, SUM_2L, SUM_L |
| dataset | Name of the object file that stores the training data | 0 | integer value | citations |
| gpu | Which gpu to use | 0 | integer value |

For _GraphSAGE_ and _GraphSAGE_RL_-based recommendation models, run:

```
python preprocess_data.py embedding_type $EMBEDDING_TYPE dataset $DATASET --threshold $THRESHOLD --gpu $GPU
```

| **Parameter** | **Description** | **Default** | **Options** | 
| :-----------: | :------------- | :----------: | :---------- |
| embedding_type | Type of SciBERT embedding | - | AVG_L, AVG_2L, AVG_SUM_ALL, AVG_SUM_L4, CONC_AVG_MAX_2L, CONC_AVG_MAX_SUM_L4, MAX_2L, SUM_2L, SUM_L |
| dataset | Name of the object file that stores the training data | - | citations, citations_authors_het_edges, authors (only _GraphSAGE_) |
| threshold | Threshold for edge weights in heterogeneous graph | 2| integer value |
| gpu | Which gpu to use | 0 | integer value | 

For _GAT_-based recommendation models, run:

```
python gat_preprocess_data.py embedding_type $EMBEDDING_TYPE dataset $DATASET --graph_type $GRAPH_TYPE --threshold $THRESHOLD --gpu $GPU
```

| **Parameter** | **Description** | **Default** | **Options** | 
| :-----------: | :------------- | :----------: | :---------- |
| embedding_type | Type of SciBERT embedding | - | AVG_L, AVG_2L, AVG_SUM_ALL, AVG_SUM_L4, CONC_AVG_MAX_2L, CONC_AVG_MAX_SUM_L4, MAX_2L, SUM_2L, SUM_L |
| dataset | Name of the object file that stores the training data | - | citations, citations_authors_het_edges |
| graph_type | The type of graph used | directed | directed, undirected | 
| threshold | Threshold for edge weights in heterogeneous graph | 2| integer value |
| gpu | Which gpu to use | 0 | integer value |


For _HAN_-based recommendation models, run:

```
python han_preprocess_data.py embedding_type $EMBEDDING_TYPE --gpu $GPU
```

| **Parameter** | **Description** | **Default** | **Options** | 
| :-----------: | :------------- | :----------: | :---------- |
| embedding_type | Type of SciBERT embedding | - | AVG_L, AVG_2L, AVG_SUM_ALL, AVG_SUM_L4, CONC_AVG_MAX_2L, CONC_AVG_MAX_SUM_L4, MAX_2L, SUM_2L, SUM_L |
| gpu | Which gpu to use | 0 | integer value |

To preprocess the training data for the ARGA models, first preprocess data for the _GAT_-based recommendation models.

For _SciBERT + ARGA_-based recommendation models, run:

```
python preprocess_data_scibert_arga.py embedding_type $EMBEDDING_TYPE dataset $DATASET --arga_model_name $ARGA_MODEL_NAME --graph_type $GRAPH_TYPE --mode $MODE --n_latent $N_LATENT --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY --dropout $DROPOUT --epochs $EPOCHS --gpu $GPU
```

| **Parameter** | **Description** | **Default** | **Options** | **Mandatory |
| :-----------: | :------------- | :----------: | :---------- | :---------: |
| embedding_type | Type of SciBERT embedding | - | AVG_L, AVG_2L, AVG_SUM_ALL, AVG_SUM_L4, CONC_AVG_MAX_2L, CONC_AVG_MAX_SUM_L4, MAX_2L, SUM_2L, SUM_L | Yes |
| dataset | Name of the object file that stores the training data | - | citations, citations_authors_het_edges | Yes |
| arga_model_name | The ARGA model used | - | ARGA, ARGVA | Yes |
| graph_type | The type of graph used | directed | directed, undirected | No |
| mode | Whether to set the ARGA net to training mode | train | train, test | No |
| n_latent | Number of units in ARGA hidden layer | 16 | integer value | No |
| learning_rate | Initial learning rate | 0.001 | float value | No |
| weight_decay | Weight for L2 loss on embedding matrix | 0 | float value | No |
| dropout | Dropout rate (1 - keep probability) | 0 | float value | No |
| epochs | Number of epochs for the ARGA model | 200 | integer value | No |
| gpu | Which gpu to use | 0 | integer value | No |
 

## Training 


## Evaluating on the test set


## Querying

Single or batch queries can be run using the  _query_single_ and _query_batch_ methods in the `<model_name>Model.py` of each model.
