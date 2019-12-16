import os
import sys
import json
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.linalg import qr
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse.linalg import norm as sparsenorm
from networkx.readwrite import json_graph

from lanczos import lanczos
from sparse_tensor_utils import *

# DISCLAIMER:
# This code file is derived from https://github.com/huangwb/AS-GCN


def parse_index_file(filename):
    """Parse index file."""
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
    """Load data."""
    print("Loading data...")
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

    print("Loaded.\n")
    print("Adjacency matrix shape: {}.".format(adj.shape))
    print("Features matrix shape: {}.".format(features.shape))

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


def nontuple_preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def nontuple_preprocess_adj(adj):
    adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)
    return adj_normalized.tocsr()


def column_prop(adj):
    column_norm = sparsenorm(adj, axis=0)
    norm_sum = sum(column_norm)
    return column_norm/norm_sum


def mix_prop(adj, features, sparseinputs=False):
    adj_column_norm = sparsenorm(adj, axis=0)
    if sparseinputs:
        features_row_norm = sparsenorm(features, axis=1)
    else:
        features_row_norm = np.linalg.norm(features, axis=1)
    mix_norm = adj_column_norm*features_row_norm

    norm_sum = sum(mix_norm)
    return mix_norm / norm_sum


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion
    to tuple representation."""
    adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)
    return sparse_to_tuple(adj_normalized)


def dense_lanczos(A, K):
    q = np.random.randn(A.shape[0], )
    Q, sigma = lanczos(A, K, q)
    A2 = np.dot(Q[:, :K], np.dot(sigma[:K, :K], Q[:, :K].T))
    return sp.csr_matrix(A2)


def sparse_lanczos(A, k):
    q = sp.random(A.shape[0], 1)
    n = A.shape[0]
    Q = sp.lil_matrix(np.zeros((n, k+1)))
    A = sp.lil_matrix(A)
    Q[:, 0] = q/sparsenorm(q)

    alpha = 0
    beta = 0

    for i in range(k):
        if i == 0:
            q = A*Q[:, i]
        else:
            q = A*Q[:, i] - beta*Q[:, i-1]
        alpha = q.T * Q[:, i]
        q = q - Q[:, i] * alpha
        q = q - Q[:, :i] * Q[:, :i].T * q  # full reorthogonalization
        beta = sparsenorm(q)
        Q[:, i+1] = q/beta
        print(i)
    Q = Q[:, :k]
    Sigma = Q.T * A * Q
    A2 = Q[:, :k] * Sigma[:k, :k] * Q[:, :k].T
    return A2


def dense_RandomSVD(A, K):
    G = np.random.randn(A.shape[0], K)
    B = np.dot(A, G)
    Q, R = qr(B, mode='economic')
    M = np.dot(Q, np.dot(Q.T, A))
    return sp.csr_matrix(M)


def construct_feed_dict(features, supports, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: supports[i] for i in range(
            len(supports))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def construct_feed_dict_with_prob(features_inputs, supports, probs, labels,
                                  labels_mask, placeholders):
    """Construct feed dictionary with adding sampling prob."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features_inputs'][i]: features_inputs[i]
                     for i in range(len(features_inputs))})
    feed_dict.update({placeholders['support'][i]: supports[i] for i in range(
            len(supports))})
    feed_dict.update({placeholders['prob'][i]: probs[i] for i in range(
            len(probs))})
    feed_dict.update({placeholders['num_features_nonzero']: features_inputs[
            1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k.
    Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(
            adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def iterate_minibatches_listinputs(inputs, batchsize, shuffle=False):
    assert inputs is not None
    numSamples = inputs[0].shape[0]
    indices = np.arange(numSamples)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, numSamples - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [input[excerpt] for input in inputs]


def transferLabel2Onehot(labels, N):
    y = np.zeros((len(labels), N))
    for i in range(len(labels)):
        pos = labels[i]
        y[i, pos] = 1
    return y


def construct_feeddict_forMixlayers(AXfeatures, support, labels, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['AXfeatures']: AXfeatures})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['num_features_nonzero']: AXfeatures[1].shape
                      })
    return feed_dict


def prepare_data(embedding_type, dataset, max_degree):
    adj, features, y_train, y_val, train_mask, val_mask = load_data(
            embedding_type, dataset)

    train_index = np.where(train_mask)[0]
    adj_train = adj[train_index, :][:, train_index]
    y_train = y_train[train_index]
    val_index = np.where(val_mask)[0]
    y_val = y_val[val_index]

    num_train = adj_train.shape[0]
    input_dim = features.shape[1]

    features = nontuple_preprocess_features(features).todense()
    train_features = features[train_index]

    norm_adj_train = nontuple_preprocess_adj(adj_train)
    norm_adj = nontuple_preprocess_adj(adj)

    adj_train, adj_val_train = compute_adjlist(norm_adj_train, max_degree)
    train_features = np.concatenate((train_features, np.zeros((1, input_dim))))

    return norm_adj, adj_train, adj_val_train, features, train_features, y_train, y_val, val_index
