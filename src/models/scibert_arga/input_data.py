import os
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

# DISCLAIMER:
# This code file is derived from https://github.com/Ruiqi-Hu/ARGA,
# which is under an identical MIT license as graph_confrec.


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(embedding_type, dataset_str):
    # load the data: x, tx, allx, graph
    path_persistent = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "..", "..", "..", "data", "interim", "gat",
                                   embedding_type, dataset_str)
    names = ['x', 'y', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open(path_persistent + "/ind.{}.{}".format(dataset_str, names[i]),
                  'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, allx, ally, graph = tuple(objects)
    print("Graph size: {}.".format(len(graph)))

    features = allx.tolil()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = ally

    idx_train = range(len(y))
    idx_val = range(len(y), len(ally))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]

    print("Adjacency matrix shape: {}.".format(adj.shape))
    print("Features matrix shape: {}.".format(features.shape))

    return adj, features, y_train, y_val, train_mask, val_mask, np.argmax(
            labels, 1)
