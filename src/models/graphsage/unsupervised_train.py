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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)
tf.compat.v1.disable_eager_execution()

# Settings
FLAGS = flags.FLAGS

flags.DEFINE_boolean('log_device_placement', False,
                     """Whether to log device placement.""")

# Core params..
flags.DEFINE_string('model', 'graphsage',
                    'model names. See README for possible values.')
flags.DEFINE_float('learning_rate', 0.00001, 'initial learning rate.')
flags.DEFINE_string("model_size", "small",
                    "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '',
                    'name of the object file that stores the training data. '
                    + 'must be specified.')

# left to default values in main experiments
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0,
                   'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 100, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of users samples in layer 2')
flags.DEFINE_integer('dim_1', 128,
                     'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128,
                     'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True,
                     'Whether to use random context or direct edges')
flags.DEFINE_integer('neg_sample_size', 20, 'number of negative samples')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_integer('identity_dim', 0,
                     'Set to positive value to use identity embedding '
                     + 'features of that dimension. Default 0.')

# Logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', True,
                     'whether to save embeddings for all nodes after training')
flags.DEFINE_string('base_log_dir', '../../../data/processed/graphsage/',
                    'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000,
                     "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256,
                     "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('print_every', 50, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10,
                     "Maximum total number of iterations")

GPU_MEM_FRACTION = 0.8


def log_dir():
    log_dir = FLAGS.base_log_dir + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val = minibatch_iter.val_feed_dict(size)
    outs_val = sess.run([model.loss, model.ranks, model.mrr],
                        feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)


def incremental_evaluate(sess, model, minibatch_iter, size):
    t_test = time.time()
    finished = False
    val_losses = []
    val_mrrs = []
    iter_num = 0
    while not finished:
        feed_dict_val, finished, _ = minibatch_iter.incremental_val_feed_dict(
                size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.ranks, model.mrr],
                            feed_dict=feed_dict_val)
        val_losses.append(outs_val[0])
        val_mrrs.append(outs_val[2])
    return np.mean(val_losses), np.mean(val_mrrs), (time.time() - t_test)


def save_val_embeddings(sess, model, minibatch_iter, size, out_dir):
    print("Saving embeddings...")
    logging.info("Saving embeddings...")
    val_embeddings = []
    finished = False
    seen = set([])
    nodes = []
    iter_num = 0
    while not finished:
        feed_dict_val, finished, edges = minibatch_iter.incremental_embed_feed_dict(
                                        size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.mrr, model.outputs1],
                            feed_dict=feed_dict_val)
        # ONLY SAVE FOR embeds1 because of planetoid
        for i, edge in enumerate(edges):
            if not edge[0] in seen:
                val_embeddings.append(outs_val[-1][i, :])
                nodes.append(edge[0])
                seen.add(edge[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    val_embeddings = np.vstack(val_embeddings)
    np.save(out_dir + "embeddings.npy", val_embeddings)
    with open(out_dir + "embeddings_ids.txt", "w") as fp:
        fp.write("\n".join(map(str, nodes)))
    print("Embeddings saved.\n")
    logging.info("Embeddings saved.")


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


def plot_losses(train_losses, validation_losses):
    # Plot the training and validation losses
    ymax = max(max(train_losses), max(validation_losses))
    ymin = min(min(train_losses), min(validation_losses))
    plt.plot(train_losses, color='tab:blue')
    plt.plot(validation_losses, color='tab:orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend(["train", "validation"], loc=3)
    plt.ylim(ymin=ymin-0.5, ymax=ymax+0.5)
    plt.savefig(log_dir() + "losses.png", bbox_inches="tight")
#    plt.show()


def print_stats(train_losses, validation_losses, training_time):
    epochs = len(train_losses)
    time_per_epoch = training_time/epochs
    epoch_min_val = validation_losses.index(min(validation_losses))

    stats_file = log_dir() + "stats.txt"
    with open(stats_file, "w") as f:
        _print("Total number of epochs trained: {}, average time per epoch: {} minutes.\n".format(
                epochs, round(time_per_epoch/60, 4)), f)
        _print("Total time trained: {} minutes.\n".format(
                round(training_time/60, 4)), f)
        _print("Lowest validation loss at epoch {} = {}.\n".format(
                epoch_min_val, validation_losses[epoch_min_val]), f)

        f.write("\nLosses:\n")
        formatting = "{:" + str(len(str(train_losses[0]))) \
                     + "d}: {:13.10f} {:13.10f}\n"
        for epoch in range(epochs):
            f.write(formatting.format(epoch+1, train_losses[epoch],
                                      validation_losses[epoch]))


def _print(text, f):
    print(text)
    f.write(text)


def train(train_data, test_data=None):
    timer = Timer()
    timer.tic()

    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]

    if features is not None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
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
        # Create model
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
        # Create model
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
    saver = tf.compat.v1.train.Saver(max_to_keep=FLAGS.epochs)

    # Init variables
    sess.run(tf.compat.v1.global_variables_initializer(),
             feed_dict={adj_info_ph: minibatch.adj})

    # Train model
    train_shadow_mrr = None
    shadow_mrr = None

    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_losses = []
    validation_losses = []

    train_adj_info = tf.compat.v1.assign(adj_info, minibatch.adj)
    val_adj_info = tf.compat.v1.assign(adj_info, minibatch.test_adj)

    for epoch in range(FLAGS.epochs):
        minibatch.shuffle()

        iter = 0
        print('Epoch: %04d' % (epoch))
        logging.info('Epoch: %04d' % (epoch))
        epoch_val_costs.append(0)
        train_loss_epoch = []
        validation_loss_epoch = []
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            # Training step
            outs = sess.run([merged, model.opt_op, model.loss, model.ranks,
                             model.aff_all, model.mrr, model.outputs1],
                            feed_dict=feed_dict)

            train_cost = outs[2]
            train_mrr = outs[5]
            train_loss_epoch.append(train_cost)
            if train_shadow_mrr is None:
                train_shadow_mrr = train_mrr
            else:
                train_shadow_mrr -= (1-0.99) * (train_shadow_mrr - train_mrr)

            if iter % FLAGS.validate_iter == 0:
                # Validation
                sess.run(val_adj_info.op)
                val_cost, ranks, val_mrr, duration = evaluate(
                        sess, model, minibatch, size=FLAGS.validate_batch_size)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost
                validation_loss_epoch.append(val_cost)
            if shadow_mrr is None:
                shadow_mrr = val_mrr
            else:
                shadow_mrr -= (1-0.99) * (shadow_mrr - val_mrr)

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)

            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (
                    total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                print("Iter: %04d" % iter,
                      "train_loss={:.5f}".format(train_cost),
                      "train_mrr={:.5f}".format(train_mrr),
                      # exponential moving average
                      "train_mrr_ema={:.5f}".format(train_shadow_mrr),
                      "val_loss={:.5f}".format(val_cost),
                      "val_mrr={:.5f}".format(val_mrr),
                      # exponential moving average
                      "val_mrr_ema={:.5f}".format(shadow_mrr),
                      "time={:.5f}".format(avg_time))
                logging.info("Iter: %04d" % iter + " " +
                             "train_loss={:.5f}".format(train_cost) + " " +
                             "train_mrr={:.5f}".format(train_mrr) + " " +
                             # exponential moving average
                             "train_mrr_ema={:.5f}".format(train_shadow_mrr) +
                             " " + "val_loss={:.5f}".format(val_cost) + " " +
                             "val_mrr={:.5f}".format(val_mrr) + " " +
                             # exponential moving average
                             "val_mrr_ema={:.5f}".format(shadow_mrr) + " " +
                             "time={:.5f}".format(avg_time))

            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        # Keep track of train and validation losses per epoch
        train_losses.append(sum(train_loss_epoch)/len(train_loss_epoch))
        validation_losses.append(
                sum(validation_loss_epoch)/len(validation_loss_epoch))

        # Save embeddings if the epoch has the lowest validation loss so far
        if FLAGS.save_embeddings and validation_losses[-1] == min(
                validation_losses):
            print("Minimum validation loss so far ({}) at epoch {}.".format(
                    current_loss, epoch))
            logging.info(
                    "Minimum validation loss so far ({}) at epoch {}.".format(
                            current_loss, epoch))
            sess.run(val_adj_info.op)
            save_val_embeddings(sess, model, minibatch,
                                FLAGS.validate_batch_size, log_dir())

        # Save model at each epoch
        print("Saving model at epoch {}.".format(epoch))
        logging.info("Saving model at epoch {}.".format(epoch))
        saver.save(sess, os.path.join(log_dir(), "model_epoch_" + str(epoch)
                   + ".ckpt"), global_step=total_steps)

        if total_steps > FLAGS.max_total_steps:
            break

    print("Optimization finished!\n")
    logging.info("Optimization finished.")

    training_time = timer.toc()
    plot_losses(train_losses, validation_losses)
    print_stats(train_losses, validation_losses, training_time)


def _set_logging():
    logging_file = log_dir() + "model.log"
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=logging_file, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)


def main(argv=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    _set_logging()
    logging.info("Starting...")
    logging.info("Loading training data..")
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix, load_walks=True)
    print("Done loading training data..\n")
    logging.info("Done loading training data.")
    print("Training model...")
    logging.info("Training model...")
    train(train_data)
    logging.info("Finished.")


if __name__ == '__main__':
    app.run(main)
