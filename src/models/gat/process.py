# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import tensorflow as tf
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh

# DISCLAIMER:
# This code file is derived from https://github.com/PetarV-/GAT,
# which is under an identical MIT license as graph_confrec.
# Conversion of original code to TF 2.0 is inspired by
# https://github.com/calciver/Graph-Attention-Networks/blob/master/Tensorflow_2_0_Graph_Attention_Networks_(GAT).ipynb


"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)

###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(embedding_type, dataset_str):
    """Load data."""
    path_persistent = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "..", "..", "..", "data", "interim", "gat",
                                   embedding_type)
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

    labels = np.vstack((ally, ty))

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
    print("Training data: {}.".format(len(train_mask[np.where(
            train_mask is True)])))
    print("Validation data: {}.".format(len(val_mask[np.where(
            val_mask is True)])))

    return adj, features, y_train, y_val, train_mask, val_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask is True, :].mean(axis=0)
    sigma = f[train_mask is True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask is True, :].mean(axis=0)
    sigma = f[train_mask is True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and
        conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return tf.SparseTensor(indices=indices, values=adj.data,
                           dense_shape=adj.shape)
