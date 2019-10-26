from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
from absl import app
from absl import flags
import pickle

from models import SampleAndAggregate, SAGEInfo
from minibatch import EdgeMinibatchIterator
from neigh_samplers import UniformNeighborSampler
from utils import load_data

import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))
from TimerCounter import Timer


# DISCLAIMER:
# This code file is derived from https://github.com/williamleif/GraphSAGE,
# which is under an identical MIT license as graph_confrec.




def log_dir():
    log_dir = FLAGS.base_log_dir + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def construct_placeholders():
    # Define placeholders
    placeholders = {
        'batch1': tf.compat.v1.placeholder(tf.int32, shape=(None),
                                           name='batch1'),
        'batch2': tf.compat.v1.placeholder(tf.int32, shape=(None),
                                           name='batch2'),
        # negative samples for all nodes in the batch
        'neg_samples': tf.compat.v1.placeholder(tf.int32, shape=(None,),
                                                name='neg_sample_size'),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=(),
                                                         name='dropout'),
        'batch_size': tf.compat.v1.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders


def infer_embeddings(test_data):
    timer = Timer()
    timer.tic()

    G = test_data[0]
    features = test_data[1]
    id_map = test_data[2]

    if features is not None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = test_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders()
    minibatch = EdgeMinibatchIterator(
            G,
            id_map,
            placeholders,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree,
            num_neg_samples=FLAGS.neg_sample_size,
            context_pairs=context_pairs)
    adj_info_ph = tf.compat.v1.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    if FLAGS.model == 'graphsage_mean':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   model_size=FLAGS.model_size,
                                   identity_dim=FLAGS.identity_dim,
                                   logging=True)
    elif FLAGS.model == 'gcn':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler,
                                FLAGS.samples_1, 2*FLAGS.dim_1),
                       SAGEInfo("node", sampler,
                                FLAGS.samples_2, 2*FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   aggregator_type="gcn",
                                   model_size=FLAGS.model_size,
                                   identity_dim=FLAGS.identity_dim,
                                   concat=False,
                                   logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   identity_dim=FLAGS.identity_dim,
                                   aggregator_type="seq",
                                   model_size=FLAGS.model_size,
                                   logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   aggregator_type="maxpool",
                                   model_size=FLAGS.model_size,
                                   identity_dim=FLAGS.identity_dim,
                                   logging=True)
    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   aggregator_type="meanpool",
                                   model_size=FLAGS.model_size,
                                   identity_dim=FLAGS.identity_dim,
                                   logging=True)

    else:
        logging.error('Error: model name unrecognized.')
        raise Exception('Error: model name unrecognized.')

    config = tf.compat.v1.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.compat.v1.Session(config=config)
    merged = tf.compat.v1.summary.merge_all()
    summary_writer = tf.compat.v1.summary.FileWriter(log_dir(), sess.graph)

    # Initialize model saver
    saver = tf.compat.v1.train.Saver()

    # Init variables
    sess.run(tf.compat.v1.global_variables_initializer(),
             feed_dict={adj_info_ph: minibatch.adj})

    train_adj_info = tf.compat.v1.assign(adj_info, minibatch.adj)
    val_adj_info = tf.compat.v1.assign(adj_info, minibatch.test_adj)

    # Restore model
    print("Restoring trained model.")
    logging.info("Restoring trained model.")
    checkpoint_file = os.path.join(log_dir(), FLAGS.model_checkpoint)
    ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_file)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored.")
    logging.info("Model restored.")

    # Compute new embeddings using restored model
    sess.run(val_adj_info.op)
    print("Computing embeddings...")
    logging.info("Computing embeddings...")
    val_embeddings = []
    finished = False
    seen = set([])
    nodes = []
    iter_num = 0
    while not finished:
        feed_dict_val, finished, edges = minibatch.incremental_embed_feed_dict(
                                        FLAGS.validate_batch_size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.mrr, model.outputs1],
                            feed_dict=feed_dict_val)
        for i, edge in enumerate(edges):
            if not edge[0] in seen:
                val_embeddings.append(outs_val[-1][i, :])
                nodes.append(edge[0])
                seen.add(edge[0])
    val_embeddings = np.vstack(val_embeddings)
    if FLAGS.save_embeddings:
        print("Saving embeddings...")
        logging.info("Saving embeddings...")
        if not os.path.exists(log_dir()):
            os.makedirs(log_dir())
        np.save(log_dir() + "inferred_embeddings.npy", val_embeddings)
        with open(log_dir() + "inferred_embeddings_ids.txt", "w") as fp:
            fp.write("\n".join(map(str, nodes)))
        print("Embeddings saved.\n")
        logging.info("Embeddings saved.")
    return nodes, val_embeddings



