# Models

## Create training data

For _ASGCN_-based recommendation models, run:

```
python preprocess_data.py embedding_type $EMBEDDING_TYPE dataset $DATASET --gpu $GPU
```

| **Parameter** | **Description** | **Default** | **Options** | **Mandatory** |
| :-----------: | :------------- | :----------: | :---------- | :---------: |
| embedding_type | Type of SciBERT embedding | - | AVG_L, AVG_2L, AVG_SUM_ALL, AVG_SUM_L4, CONC_AVG_MAX_2L, CONC_AVG_MAX_SUM_L4, MAX_2L, SUM_2L, SUM_L | Yes |
| dataset | Name of the object file that stores the training data | 0 | integer value | citations | Yes |
| gpu | Which gpu to use | 0 | integer value | No |

For _GraphSAGE_ and _GraphSAGE_RL_-based recommendation models, run:

```
python preprocess_data.py embedding_type $EMBEDDING_TYPE dataset $DATASET --threshold $THRESHOLD --gpu $GPU
```

| **Parameter** | **Description** | **Default** | **Options** | **Mandatory** |
| :-----------: | :------------- | :----------: | :---------- | :---------: |
| embedding_type | Type of SciBERT embedding | - | AVG_L, AVG_2L, AVG_SUM_ALL, AVG_SUM_L4, CONC_AVG_MAX_2L, CONC_AVG_MAX_SUM_L4, MAX_2L, SUM_2L, SUM_L | Yes |
| dataset | Name of the object file that stores the training data | - | citations, citations_authors_het_edges, authors (only _GraphSAGE_) | Yes |
| threshold | Threshold for edge weights in heterogeneous graph | 2| integer value | No |
| gpu | Which gpu to use | 0 | integer value | No |

For _GAT_-based recommendation models, run:

```
python gat_preprocess_data.py embedding_type $EMBEDDING_TYPE dataset $DATASET --graph_type $GRAPH_TYPE --threshold $THRESHOLD --gpu $GPU
```

| **Parameter** | **Description** | **Default** | **Options** | **Mandatory** |
| :-----------: | :------------- | :----------: | :---------- | :---------: |
| embedding_type | Type of SciBERT embedding | - | AVG_L, AVG_2L, AVG_SUM_ALL, AVG_SUM_L4, CONC_AVG_MAX_2L, CONC_AVG_MAX_SUM_L4, MAX_2L, SUM_2L, SUM_L | Yes |
| dataset | Name of the object file that stores the training data | - | citations, citations_authors_het_edges | Yes |
| graph_type | The type of graph used | directed | directed, undirected | No |
| threshold | Threshold for edge weights in heterogeneous graph | 2| integer value | No |
| gpu | Which gpu to use | 0 | integer value | No |


For _HAN_-based recommendation models, run:

```
python han_preprocess_data.py embedding_type $EMBEDDING_TYPE --gpu $GPU
```

| **Parameter** | **Description** | **Default** | **Options** | **Mandatory** |
| :-----------: | :------------- | :----------: | :---------- | :---------: |
| embedding_type | Type of SciBERT embedding | - | AVG_L, AVG_2L, AVG_SUM_ALL, AVG_SUM_L4, CONC_AVG_MAX_2L, CONC_AVG_MAX_SUM_L4, MAX_2L, SUM_2L, SUM_L | Yes |
| gpu | Which gpu to use | 0 | integer value | No |

To preprocess the training data for the ARGA models, first preprocess data for the _GAT_-based recommendation models.

For _SciBERT + ARGA_-based recommendation models, run:

```
python preprocess_data_scibert_arga.py embedding_type $EMBEDDING_TYPE dataset $DATASET --arga_model_name $ARGA_MODEL_NAME --graph_type $GRAPH_TYPE --mode $MODE --n_latent $N_LATENT --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY --dropout $DROPOUT --epochs $EPOCHS --gpu $GPU
```

| **Parameter** | **Description** | **Default** | **Options** | **Mandatory** |
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

For _ASGCN_-based recommendation models, run:

```
python model.py embedding_type $EMBEDDING_TYPE dataset #DATASET model_name #MODEL_NAME --max_degree $MAX_DEGREE --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY --dropout #DROPOUT --epochs $EPOCHS --early_stopping $EARLY_STOPPING --hidden1 $HIDDEN1 --rank $RANK --var $VAR --sampler_device $SAMPLER_DEVICE --gpu $GPU
```

| **Parameter** | **Description** | **Default** | **Options** | **Mandatory** |
| :-----------: | :------------- | :----------: | :---------- | :---------: |
| embedding_type | Type of SciBERT embedding | - | AVG_L, AVG_2L, AVG_SUM_ALL, AVG_SUM_L4, CONC_AVG_MAX_2L, CONC_AVG_MAX_SUM_L4, MAX_2L, SUM_2L, SUM_L | Yes |
| dataset | Name of the object file that stores the training data | - | citations, citations_authors_het_edges | Yes |
| model_name | Model names | - | gcn_adapt, gcn_adapt_mix | Yes |
| max_degree | Maximum degree for constructing the adjacency matrix | 696 | integer value | No |
| learning_rate | Initial learning rate | 0.001 | float value | No |
| weight_decay | Weight decay | 5e-4 | float value | No |
| dropout | Dropout rate (1 - keep probability) | 0 | float value | No |
| epochs | Number of epochs to train | 300 | integer value | No |
| early_stopping | Tolerance for early stopping (# of epochs) | 30 | integer value | No |
| hidden1 | Number of units in hidden layer 1 | 16 | integer value | No |
| rank | The number of nodes per layer | 128 | integer value | No |
| var | Whether to use variance reduction | 0.5 | float value | No |
| sampler_device | The device for sampling | cpu | cpu, gpu | No |
| gpu | Which gpu to use | 0 | integer value | No |

To train an _unsupervised GraphSAGE_ GNN model, run:

```
python unsupervised_model.py train_prefix #TRAIN_PREFIX model_name $MODEL_NAME --model_size $MODEL_SIZE --learning_rate $LEARNING_RATE --epochs $EPOCHS --dropout $DROPOUT --weight_decay $WEIGHT_DECAY --max_degree $MAX_DEGREE --samples_1 $SAMPLES_1 --samples_2 $SAMPLES_2 --dim_1 $DIM_1 --dim_2 $DIM_2 --neg_sample_size $NEG_SAMPLE_SIZE --batch_size $BATCH_SIZE --save_embeddings #SAVE_EMBEDDINGS --validate_iter $VALIDATE_ITER --validate_batch_size $VALIDATE_BATCH_SIZE --gpu $GPU --print_every $PRINT_EVERY --max_total_steps $MAX_TOTAL_STEPS
```

| **Parameter** | **Description** | **Default** | **Options** | **Mandatory** |
| :-----------: | :------------- | :----------: | :---------- | :---------: |
| train_prefix | Name of the object file that stores the training data | - |  | Yes |
| model_name | Model names | - | graphsage_mean, gcn, graphsage_seq, graphsage_maxpool, graphsage_meanpool | Yes |
| model_size | Model specific definitions | small | small, big | No |
| learning_rate | Initial learning rate | 0.00001 | float value | No |
| epochs | Number of epochs to train | 10 | integer value | No |
| dropout | Dropout rate (1 - keep probability) | 0 | float value | No |
| weight_decay | Weight for l2 loss on embedding matrix. | 0 | float value | No |
| max_degree | Maximum node degree | 100 | integer value | No |
| samples_1 | Number of samples in layer 1 | 25 | integer value | No |
| samples_2 | Number of samples in layer 2 | 10 | integer value | No |
| dim_1 | 'Size of output dim (final is 2x this, if using concat) | 128 | integer value | No |
| dim_2 | 'Size of output dim (final is 2x this, if using concat) | 128 | integer value | No |
| neg_sample_size | Number of negative samples | 20 | integer value | No |
| batch_size | Minibatch size | 512 | integer value | No |
| save_embeddings | Whether to save embeddings for all nodes after training | True | boolean | No |
| validate_iter | How often to run a validation minibatch | 5000 | integer value | No |
| validate_batch_size | How many nodes per validation sample | 256 | integer value | No |
| gpu | Which gpu to use | 0 | integer value | No |
| print_every | How often to print training info | 50 | integer value | No |
| max_total_steps | Maximum total number of iterations | 10<sup>10</sup>  | integer value | No |


To train a _supervised GraphSAGE_ GNN model, run:

```
python supervised_model.py train_prefix #TRAIN_PREFIX model_name $MODEL_NAME --model_size $MODEL_SIZE --learning_rate $LEARNING_RATE --epochs $EPOCHS --dropout $DROPOUT --weight_decay $WEIGHT_DECAY --max_degree $MAX_DEGREE --samples_1 $SAMPLES_1 --samples_2 $SAMPLES_2 --samples_3 $SAMPLES_3 --dim_1 $DIM_1 --dim_2 $DIM_2 --neg_sample_size $NEG_SAMPLE_SIZE --batch_size $BATCH_SIZE --validate_iter $VALIDATE_ITER --validate_batch_size $VALIDATE_BATCH_SIZE --gpu $GPU --print_every $PRINT_EVERY --max_total_steps $MAX_TOTAL_STEPS
```

| **Parameter** | **Description** | **Default** | **Options** | **Mandatory** |
| :-----------: | :------------- | :----------: | :---------- | :---------: |
| train_prefix | Name of the object file that stores the training data | - |  | Yes |
| model_name | Model names | - | graphsage_mean, gcn, graphsage_seq, graphsage_maxpool, graphsage_meanpool | Yes |
| model_size | Model specific definitions | small | small, big | No |
| learning_rate | Initial learning rate | 0.001 | float value | No |
| epochs | Number of epochs to train | 10 | integer value | No |
| dropout | Dropout rate (1 - keep probability) | 0 | float value | No |
| weight_decay | Weight for l2 loss on embedding matrix. | 0 | float value | No |
| max_degree | Maximum node degree | 100 | integer value | No |
| samples_1 | Number of samples in layer 1 | 25 | integer value | No |
| samples_2 | Number of samples in layer 2 | 10 | integer value | No |
| samples_3 | Number of samples in layer 3 (Only for mean model) | 0 | integer value | No |
| dim_1 | 'Size of output dim (final is 2x this, if using concat) | 128 | integer value | No |
| dim_2 | 'Size of output dim (final is 2x this, if using concat) | 128 | integer value | No |
| neg_sample_size | Number of negative samples | 20 | integer value | No |
| batch_size | Minibatch size | 512 | integer value | No |
| validate_iter | How often to run a validation minibatch | 5000 | integer value | No |
| validate_batch_size | How many nodes per validation sample | 256 | integer value | No |
| gpu | Which gpu to use | 0 | integer value | No |
| print_every | How often to print training info | 5 | integer value | No |
| max_total_steps | Maximum total number of iterations | 10<sup>10</sup>  | integer value | No |



To train an _unsupervised GraphSAGE_RL_ GNN model, run:

```
python unsupervised_model.py train_prefix #TRAIN_PREFIX model_name $MODEL_NAME --nonlinear_sampler $NONLINEAR_SAMPLER --uniform_ratio $UNIFORM_RATIO --model_size $MODEL_SIZE --learning_rate $LEARNING_RATE --epochs $EPOCHS --dropout $DROPOUT --weight_decay $WEIGHT_DECAY --max_degree $MAX_DEGREE --samples_1 $SAMPLES_1 --samples_2 $SAMPLES_2 --dim_1 $DIM_1 --dim_2 $DIM_2 --neg_sample_size $NEG_SAMPLE_SIZE --batch_size $BATCH_SIZE --save_embeddings #SAVE_EMBEDDINGS --validate_iter $VALIDATE_ITER --validate_batch_size $VALIDATE_BATCH_SIZE --gpu $GPU --print_every $PRINT_EVERY --max_total_steps $MAX_TOTAL_STEPS
```

| **Parameter** | **Description** | **Default** | **Options** | **Mandatory** |
| :-----------: | :------------- | :----------: | :---------- | :---------: |
| train_prefix | Name of the object file that stores the training data | - |  | Yes |
| model_name | Model names | - | mean_concat, mean_add, gcn, graphsage_seq, graphsage_maxpool, graphsage_meanpool | Yes |
| nonlinear_sampler | Where to use nonlinear sampler o.w. linear sampler | False | boolean | No |
| uniform_ratio | In case of FastML sampling, the percentile of uniform sampling preceding the regressor sampling | 0.6 | float value | No |
| model_size | Model specific definitions | small | small, big | No |
| learning_rate | Initial learning rate | 0.00001 | float value | No |
| epochs | Number of epochs to train | 10 | integer value | No |
| dropout | Dropout rate (1 - keep probability) | 0 | float value | No |
| weight_decay | Weight for l2 loss on embedding matrix. | 0 | float value | No |
| max_degree | Maximum node degree | 100 | integer value | No |
| samples_1 | Number of samples in layer 1 | 25 | integer value | No |
| samples_2 | Number of samples in layer 2 | 10 | integer value | No |
| dim_1 | 'Size of output dim (final is 2x this, if using concat) | 128 | integer value | No |
| dim_2 | 'Size of output dim (final is 2x this, if using concat) | 128 | integer value | No |
| neg_sample_size | Number of negative samples | 20 | integer value | No |
| batch_size | Minibatch size | 512 | integer value | No |
| save_embeddings | Whether to save embeddings for all nodes after training | True | boolean | No |
| validate_iter | How often to run a validation minibatch | 5000 | integer value | No |
| validate_batch_size | How many nodes per validation sample | 512 | integer value | No |
| gpu | Which gpu to use | 0 | integer value | No |
| print_every | How often to print training info | 50 | integer value | No |
| max_total_steps | Maximum total number of iterations | 10<sup>10</sup>  | integer value | No |



To train a _supervised GraphSAGE_RL_ GNN model, run:

```
python supervised_model.py train_prefix #TRAIN_PREFIX model_name $MODEL_NAME --nonlinear_sampler $NONLINEAR_SAMPLER --allhop_rewards $ALLHOP_REWARDS --model_size $MODEL_SIZE --learning_rate $LEARNING_RATE --epochs $EPOCHS --dropout $DROPOUT --weight_decay $WEIGHT_DECAY --max_degree $MAX_DEGREE --samples_1 $SAMPLES_1 --samples_2 $SAMPLES_2 --samples_3 $SAMPLES_3 --dim_1 $DIM_1 --dim_2 $DIM_2 --dim_3 $DIM_3 --neg_sample_size $NEG_SAMPLE_SIZE --batch_size $BATCH_SIZE --save_embeddings #SAVE_EMBEDDINGS --validate_iter $VALIDATE_ITER --validate_batch_size $VALIDATE_BATCH_SIZE --gpu $GPU --print_every $PRINT_EVERY --max_total_steps $MAX_TOTAL_STEPS
```

| **Parameter** | **Description** | **Default** | **Options** | **Mandatory** |
| :-----------: | :------------- | :----------: | :---------- | :---------: |
| train_prefix | Name of the object file that stores the training data | - |  | Yes |
| model_name | Model names | - | mean_concat, mean_add, gcn, graphsage_seq, graphsage_maxpool, graphsage_meanpool | Yes |
| nonlinear_sampler | Where to use nonlinear sampler o.w. linear sampler | True | boolean | No |
| allhop_rewards | Whether to use a all-hop rewards or last-hop reward for training the nonlinear sampler | False | boolean | No |
| model_size | Model specific definitions | small | small, big | No |
| learning_rate | Initial learning rate | 0.001 | float value | No |
| epochs | Number of epochs to train | 10 | integer value | No |
| dropout | Dropout rate (1 - keep probability) | 0 | float value | No |
| weight_decay | Weight for l2 loss on embedding matrix. | 0 | float value | No |
| max_degree | Maximum node degree | 100 | integer value | No |
| samples_1 | Number of samples in layer 1 | 25 | integer value | No |
| samples_2 | Number of samples in layer 2 | 10 | integer value | No |
| samples_3 | Number of samples in layer 3 (Only for mean model) | 0 | integer value | No |
| dim_1 | 'Size of output dim (final is 2x this, if using concat) | 512 | integer value | No |
| dim_2 | 'Size of output dim (final is 2x this, if using concat) | 512 | integer value | No |
| dim_2 | 'Size of output dim (final is 2x this, if using concat) | 0 | integer value | No |
| neg_sample_size | Number of negative samples | 20 | integer value | No |
| batch_size | Minibatch size | 128 | integer value | No |
| validate_iter | How often to run a validation minibatch | 5000 | integer value | No |
| validate_batch_size | How many nodes per validation sample | 128 | integer value | No |
| gpu | Which gpu to use | 0 | integer value | No |
| print_every | How often to print training info | 5 | integer value | No |
| max_total_steps | Maximum total number of iterations | 10<sup>10</sup>  | integer value | No |











For _GAT_-based recommendation models, run:

```
python train_gat.py embedding_type $EMBEDDING_TYPE dataset #DATASET --graph_type $GRAPH_TYPE --hid_units #HID_UNITS --n_heads $N_HEADS --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY --epochs $EPOCHS --patience $PATIENCE --residual $RESIDUAL --ffd_drop $FFD_DROP --attn_drop $ATTN_DROP --gpu $GPU
```

| **Parameter** | **Description** | **Default** | **Options** | **Mandatory** |
| :-----------: | :------------- | :----------: | :---------- | :---------: |
| embedding_type | Type of SciBERT embedding | - | AVG_L, AVG_2L, AVG_SUM_ALL, AVG_SUM_L4, CONC_AVG_MAX_2L, CONC_AVG_MAX_SUM_L4, MAX_2L, SUM_2L, SUM_L | Yes |
| dataset | Name of the object file that stores the training data | - | citations | Yes |
| graph_type | The type of graph used | directed | directed, undirected | No |
| hid_units | Number of hidden units per each attention head in each layer | [64] | (multiple) integer values | No |
| n_heads | Additional entry for the output layer | [8, 1] | (multiple) integer values | No |
| learning_rate | Learning rate | 0.005 | float value | No |
| weight_decay | Weight decay | 0 | float value | No |
| epochs | Number of epochs to train | 100000 | integer value | No |
| early_stopping | Tolerance for early stopping (# of epochs) | 100 | integer value | No |
| residual | Whether to include residual connections | False | boolean | No |
| ffd_drop | Dropout rate layers' inputs | 0.5 | float value | No |
| attn_drop | Dropout rate for the normalized attention coefficients | 0.5 | float value | No |
| gpu | Which gpu to use | None | integer value | No |

For _HAN_-based recommendation models, run:

```
python train_han.py model $MODEL embedding_type $EMBEDDING_TYPE --hid_units #HID_UNITS --n_heads $N_HEADS --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY --epochs $EPOCHS --patience $PATIENCE --residual $RESIDUAL --ffd_drop $FFD_DROP --attn_drop $ATTN_DROP --gpu $GPU
```

| **Parameter** | **Description** | **Default** | **Options** | **Mandatory** |
| :-----------: | :------------- | :----------: | :---------- | :---------: |
| model | The type of model used | - | HeteGAT, HeteGAT_multi | Yes |
| embedding_type | Type of SciBERT embedding | - | AVG_L, AVG_2L, AVG_SUM_ALL, AVG_SUM_L4, CONC_AVG_MAX_2L, CONC_AVG_MAX_SUM_L4, MAX_2L, SUM_2L, SUM_L | Yes |
| hid_units | Number of hidden units per each attention head in each layer | [64] | (multiple) integer values | No |
| n_heads | Additional entry for the output layer | [8, 1] | (multiple) integer values | No |
| learning_rate | Learning rate | 0.005 | float value | No |
| weight_decay | Weight decay | 0 | float value | No |
| epochs | Number of epochs to train | 10000 | integer value | No |
| early_stopping | Tolerance for early stopping (# of epochs) | 100 | integer value | No |
| residual | Whether to include residual connections | False | boolean | No |
| ffd_drop | Dropout rate layers' inputs | 0.5 | float value | No |
| attn_drop | Dropout rate for the normalized attention coefficients | 0.5 | float value | No |
| gpu | Which gpu to use | None | integer value | No |

To train the _ARGA_ GNN model, run:
```
python arga.py embedding_type $EMBEDDING_TYPE dataset #DATASET model_name $MODEL_NAME model $MODEL --mode $MODE --n_latent $N_LATENT --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY --dropout $DROPOUT --epochs $EPOCHS --gpu $GPU

```

| **Parameter** | **Description** | **Default** | **Options** | **Mandatory** |
| :-----------: | :------------- | :----------: | :---------- | :---------: |
| embedding_type | Type of SciBERT embedding | - | AVG_L, AVG_2L, AVG_SUM_ALL, AVG_SUM_L4, CONC_AVG_MAX_2L, CONC_AVG_MAX_SUM_L4, MAX_2L, SUM_2L, SUM_L | Yes |
| dataset | Name of the object file that stores the training data | - | citations, citations_authors_het_edges | Yes |
| model_name | Type of model | - | ARGA, ARGVA | Yes |
| graph_type | The type of graph used | directed | directed, undirected | No |
| mode | Whether to set the net to training mode | train | train, test | No |
| n_latent | Number of units in ARGA hidden layer | 16 | integer value | No |
| learning_rate | Initial learning rate | 0.001 | float value | No |
| weight_decay | Weight for L2 loss on embedding matrix | 0 | float value | No |
| dropout | Dropout rate (1 - keep probability) | 0 | float value | No |
| epochs | Number of epochs for the ARGA model | 200 | integer value | No |
| gpu | Which gpu to use | None | integer value | No |


To train the FFNN of the _SciBERT + ARGA_-based recommendation model, run: 

```
python ffnn.py embedding_type $EMBEDDING_TYPE dataset #DATASET model_name $MODEL_NAME model $MODEL --mode $MODE --n_latent $N_LATENT --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY --dropout $DROPOUT --epochs $EPOCHS --gpu $GPU --ffnn_hidden_dim $FFNN_HIDDEN_DIM

```

| **Parameter** | **Description** | **Default** | **Options** | **Mandatory** |
| :-----------: | :------------- | :----------: | :---------- | :---------: |
| embedding_type | Type of SciBERT embedding | - | AVG_L, AVG_2L, AVG_SUM_ALL, AVG_SUM_L4, CONC_AVG_MAX_2L, CONC_AVG_MAX_SUM_L4, MAX_2L, SUM_2L, SUM_L | Yes |
| dataset | Name of the object file that stores the training data | - | citations, citations_authors_het_edges | Yes |
| model_name | Type of model | - | ARGA, ARGVA | Yes |
| graph_type | The type of graph used | directed | directed, undirected | No |
| mode | Whether to set the net to training mode | train | train, test | No |
| n_latent | Number of units in ARGA hidden layer | 16 | integer value | No |
| learning_rate | Initial learning rate | 0.001 | float value | No |
| weight_decay | Weight for L2 loss on embedding matrix | 0 | float value | No |
| dropout | Dropout rate (1 - keep probability) | 0 | float value | No |
| epochs | Number of epochs for the ARGA model | 200 | integer value | No |
| gpu | Which gpu to use | None | integer value | No |
| ffnn_hidden_dim | Number of units in hidden layer of the FFNN | 100 | integer value | No |


## Evaluating on the test set

All the evaluation scripts take as final parameter _recs = the number of recommendations generated (default: 10)_.

To evaluate the _ASGCN_, _GAT_, _HAN_, _SciBERT + ARGA_-based recommendation models, run: `python <model_name>ModelEvaluation.py` with the same parameters used for training the model (see above).



## Querying

Single or batch queries can be run using the  _query_single_ and _query_batch_ methods in the `<model_name>Model.py` of each model.
