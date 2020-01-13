import tensorflow as tf
import numpy as np
from model import ARGA, ARVGA, Discriminator
from optimizer import OptimizerAE, OptimizerVAE
import scipy.sparse as sp
from input_data import load_data
from preprocessing import preprocess_graph, sparse_to_tuple, mask_test_edges
from preprocessing import construct_feed_dict

# DISCLAIMER:
# This code file is derived from https://github.com/Ruiqi-Hu/ARGA,
# which is under an identical MIT license as graph_confrec.


def get_placeholder(adj, hidden2):
    placeholders = {
        'features': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'real_distribution': tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[adj.shape[0], hidden2],
                name='real_distribution')
    }
    return placeholders


def get_model(model_str, placeholders, num_features, num_nodes,
              features_nonzero, hidden1, hidden2, hidden3):
    discriminator = Discriminator(hidden1, hidden2, hidden3)
    d_real = discriminator.construct(placeholders['real_distribution'])
    model = None
    if model_str == 'arga_ae':
        model = ARGA(placeholders, num_features, features_nonzero, hidden1,
                     hidden2)
    elif model_str == 'arga_vae':
        model = ARVGA(placeholders, num_features, num_nodes, features_nonzero,
                      hidden1, hidden2)
    return d_real, discriminator, model


def format_data(embedding_type, data_name, use_features):
    # Load data
    adj, features, y_train, y_val, train_mask, val_mask, true_labels = load_data(
            embedding_type, data_name)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((
            adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges = mask_test_edges(adj)
    adj = adj_train

    if use_features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((
            adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    feas = {"adj": adj, "num_features": num_features, "num_nodes": num_nodes,
            "features_nonzero": features_nonzero, "pos_weight": pos_weight,
            "norm": norm, "adj_norm": adj_norm, "adj_label": adj_label,
            "features": features, "true_labels": true_labels,
            "train_edges": train_edges, "adj_orig": adj_orig}
    return feas


def get_optimizer(model_str, model, discriminator, placeholders, pos_weight,
                  norm, d_real, num_nodes, discriminator_learning_rate,
                  learning_rate):
    if model_str == 'arga_ae':
        d_fake = discriminator.construct(model.embeddings, reuse=True)
        opt = OptimizerAE(
                preds=model.reconstructions,
                labels=tf.reshape(tf.sparse.to_dense(
                        placeholders['adj_orig'], validate_indices=False),
                        [-1]),
                pos_weight=pos_weight,
                norm=norm,
                d_real=d_real,
                d_fake=d_fake,
                discriminator_learning_rate=discriminator_learning_rate,
                learning_rate=learning_rate)
    elif model_str == 'arga_vae':
        opt = OptimizerVAE(
                preds=model.reconstructions,
                labels=tf.reshape(tf.sparse.to_dense(
                        placeholders['adj_orig'], validate_indices=False),
                        [-1]),
                model=model, num_nodes=num_nodes,
                pos_weight=pos_weight,
                norm=norm,
                d_real=d_real,
                d_fake=discriminator.construct(model.embeddings, reuse=True),
                discriminator_learning_rate=discriminator_learning_rate,
                learning_rate=learning_rate)
    return opt


def update(model, opt, sess, adj_norm, adj_label, features, placeholders, adj,
           dropout, hidden2):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features,
                                    placeholders)
    feed_dict.update({placeholders['dropout']: dropout})

    feed_dict.update({placeholders['dropout']: 0})
    emb = sess.run(model.z_mean, feed_dict=feed_dict)

    z_real_dist = np.random.randn(adj.shape[0], hidden2)
    feed_dict.update({placeholders['real_distribution']: z_real_dist})

    for j in range(5):
        _, reconstruct_loss = sess.run([opt.opt_op, opt.cost],
                                       feed_dict=feed_dict)
    d_loss, _ = sess.run([opt.dc_loss, opt.discriminator_optimizer],
                         feed_dict=feed_dict)
    g_loss, _ = sess.run([opt.generator_loss, opt.generator_optimizer],
                         feed_dict=feed_dict)
    avg_cost = reconstruct_loss
    return emb, avg_cost
