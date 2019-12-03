from __future__ import division
from __future__ import print_function

import os
import sys
import pdb
import time
import pickle
import argparse
import numpy as np
from scipy import sparse
import tensorflow as tf
import multiprocessing as mp
import matplotlib.pyplot as plt

from models import SampleAndAggregate, SAGEInfo
from minibatch import EdgeMinibatchIterator
from neigh_samplers import UniformNeighborSampler
from neigh_samplers import MLNeighborSampler, FastMLNeighborSampler
from utils import load_data

sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))
from TimerCounter import Timer

# DISCLAIMER:
# This code file is derived from https://github.com/oj9040/GraphSAGE_RL.


class UnsupervisedModelRL:

    def __init__(self, train_prefix, model_name, nonlinear_sampler=False,
                 uniform_ratio=0.6, model_size="small", learning_rate=0.00001,
                 epochs=10, dropout=0.0, weight_decay=0.0, max_degree=100,
                 samples_1=25, samples_2=10, dim_1=128, dim_2=128,
                 random_context=True, neg_sample_size=20, batch_size=512,
                 identity_dim=0, save_embeddings=True,
                 base_log_dir='../../../data/processed/graphsage_rl/',
                 validate_iter=5000, validate_batch_size=512, gpu=0,
                 print_every=50, max_total_steps=10**10,
                 log_device_placement=False):

        self.train_prefix = train_prefix
        self.model_name = model_name
        self.nonlinear_sampler = nonlinear_sampler
        self.uniform_ratio = uniform_ratio
        self.model_size = model_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.max_degree = max_degree
        self.samples_1 = samples_1
        self.samples_2 = samples_2
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.random_context = random_context
        self.neg_sample_size = neg_sample_size
        self.batch_size = batch_size
        self.identity_dim = identity_dim
        self.save_embeddings = save_embeddings
        self.base_log_dir = base_log_dir
        self.validate_iter = validate_iter
        self.validate_batch_size = validate_batch_size
        self.gpu = gpu
        self.print_every = print_every
        self.max_total_steps = max_total_steps
        self.log_device_placement = log_device_placement

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        # Set random seed
        seed = 123
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        tf.compat.v1.disable_eager_execution()

    def _log_dir(self, sampler_model_name):
        log_dir = self.base_log_dir \
                    + self.train_prefix.rsplit("/", maxsplit=1)[-2] + "-" \
                    + sampler_model_name + "/unsupervised"
        log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
                    model=self.model_name, model_size=self.model_size,
                    lr=self.learning_rate)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def _sampler_log_dir(self):
        log_dir = self.base_log_dir \
                    + self.train_prefix.rsplit("/", maxsplit=1)[-2] + \
                    "/unsupervised"
        log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
                    model=self.model_name, model_size=self.model_size,
                    lr=self.learning_rate)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    # Define model evaluation function
    def _evaluate(self, sess, model, minibatch_iter, size=None):
        t_test = time.time()
        feed_dict_val = minibatch_iter.val_feed_dict(size)
        outs_val = sess.run([model.loss, model.ranks, model.mrr],
                            feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

    def _incremental_evaluate(self, sess, model, minibatch_iter, size):
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

    def _save_embeddings(self, sess, model, minibatch_iter, size, out_dir):
        print("Saving embeddings...")
        val_embeddings = []
        finished = False
        seen = set([])
        nodes = []
        iter_num = 0
        while not finished:
#            pdb.set_trace()
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

    def _construct_placeholders(self):
        # Define placeholders
        placeholders = {
            'batch1': tf.compat.v1.placeholder(tf.int32,
                                               shape=(self.batch_size),
                                               name='batch1'),
            'batch2': tf.compat.v1.placeholder(tf.int32,
                                               shape=(self.batch_size),
                                               name='batch2'),
            # negative samples for all nodes in the batch
            'neg_samples': tf.compat.v1.placeholder(tf.int32, shape=(None,),
                                                    name='neg_sample_size'),
            'dropout': tf.compat.v1.placeholder_with_default(0., shape=(),
                                                             name='dropout'),
            'batch_size': tf.compat.v1.placeholder(tf.int32,
                                                   name='batch_size'),
        }
        return placeholders

    def _plot_losses(self, train_losses, validation_losses, sampler_name):
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
        plt.savefig(self._log_dir(sampler_name) + "losses.png",
                    bbox_inches="tight")

    def _print_stats(self, train_losses, validation_losses, training_time,
                     sampler_name):
        epochs = len(train_losses)
        time_per_epoch = training_time/epochs
        epoch_min_val = validation_losses.index(min(validation_losses))

        stats_file = self._log_dir(sampler_name) + "stats.txt"
        with open(stats_file, "w") as f:
            self._print("Total number of epochs trained: {}, average time per epoch: {} minutes.\n".format(
                    epochs, round(time_per_epoch/60, 4)), f)
            self._print("Total time trained: {} minutes.\n".format(
                    round(training_time/60, 4)), f)
            self._print("Lowest validation loss at epoch {} = {}.\n".format(
                    epoch_min_val, validation_losses[epoch_min_val]), f)

            f.write("\nLosses:\n")
            formatting = "{:" + str(len(str(train_losses[0]))) \
                         + "d}: {:13.10f} {:13.10f}\n"
            for epoch in range(epochs):
                f.write(formatting.format(epoch, train_losses[epoch],
                                          validation_losses[epoch]))

    def _print(self, text, f):
        print(text)
        f.write(text)

    def _create_sampler(self, sampler_name, adj_info, features):
        if sampler_name == 'Uniform':
            sampler = UniformNeighborSampler(adj_info)
        elif sampler_name == 'ML':
            sampler = MLNeighborSampler(adj_info, features, self.max_degree,
                                        self.nonlinear_sampler)
        elif sampler_name == 'FastML':
            sampler = FastMLNeighborSampler(adj_info, features,
                                            self.max_degree,
                                            self.nonlinear_sampler)
        else:
            raise Exception('Error: sampler name unrecognized.')
        return sampler

    def _create_model(self, sampler_name, placeholders, features, adj_info,
                      minibatch):
        if self.model_name == 'mean_concat':
            # Create model
            sampler = self._create_sampler(sampler_name, adj_info, features)
            layer_infos = [SAGEInfo("node", sampler, self.samples_1,
                                    self.dim_1),
                           SAGEInfo("node", sampler, self.samples_2,
                                    self.dim_2)]
            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       concat=True,
                                       layer_infos=layer_infos,
                                       weight_decay=self.weight_decay,
                                       learning_rate=self.learning_rate,
                                       neg_sample_size=self.neg_sample_size,
                                       batch_size=self.batch_size,
                                       model_size=self.model_size,
                                       identity_dim=self.identity_dim,
                                       logging=True)
        elif self.model_name == 'mean_add':
            # Create model
            sampler = self._create_sampler(sampler_name, adj_info, features)
            layer_infos = [SAGEInfo("node", sampler, self.samples_1,
                                    self.dim_1),
                           SAGEInfo("node", sampler, self.samples_2,
                                    self.dim_2)]
            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       weight_decay=self.weight_decay,
                                       learning_rate=self.learning_rate,
                                       neg_sample_size=self.neg_sample_size,
                                       batch_size=self.batch_size,
                                       concat=False,
                                       model_size=self.model_size,
                                       identity_dim=self.identity_dim,
                                       logging=True)
        elif self.model_name == 'gcn':
            # Create model
            sampler = self._create_sampler(sampler_name, adj_info, features)
            layer_infos = [SAGEInfo("node", sampler, self.samples_1,
                                    2*self.dim_1),
                           SAGEInfo("node", sampler, self.samples_2,
                                    2*self.dim_2)]
            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       weight_decay=self.weight_decay,
                                       learning_rate=self.learning_rate,
                                       neg_sample_size=self.neg_sample_size,
                                       batch_size=self.batch_size,
                                       aggregator_type="gcn",
                                       model_size=self.model_size,
                                       identity_dim=self.identity_dim,
                                       concat=False,
                                       logging=True)
        elif self.model_name == 'graphsage_seq':
            # Create model
            sampler = self._create_sampler(sampler_name, adj_info, features)
            layer_infos = [SAGEInfo("node", sampler, self.samples_1,
                                    self.dim_1),
                           SAGEInfo("node", sampler, self.samples_2,
                                    self.dim_2)]
            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       weight_decay=self.weight_decay,
                                       learning_rate=self.learning_rate,
                                       neg_sample_size=self.neg_sample_size,
                                       batch_size=self.batch_size,
                                       identity_dim=self.identity_dim,
                                       aggregator_type="seq",
                                       model_size=self.model_size,
                                       logging=True)
        elif self.model_name == 'graphsage_maxpool':
            # Create model
            sampler = self._create_sampler(sampler_name, adj_info, features)
            layer_infos = [SAGEInfo("node", sampler, self.samples_1,
                                    self.dim_1),
                           SAGEInfo("node", sampler, self.samples_2,
                                    self.dim_2)]
            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       weight_decay=self.weight_decay,
                                       learning_rate=self.learning_rate,
                                       neg_sample_size=self.neg_sample_size,
                                       batch_size=self.batch_size,
                                       aggregator_type="maxpool",
                                       model_size=self.model_size,
                                       identity_dim=self.identity_dim,
                                       logging=True)
        elif self.model_name == 'graphsage_meanpool':
            # Create model
            sampler = self._create_sampler(sampler_name, adj_info, features)
            layer_infos = [SAGEInfo("node", sampler, self.samples_1,
                                    self.dim_1),
                           SAGEInfo("node", sampler, self.samples_2,
                                    self.dim_2)]
            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       weight_decay=self.weight_decay,
                                       learning_rate=self.learning_rate,
                                       neg_sample_size=self.neg_sample_size,
                                       batch_size=self.batch_size,
                                       aggregator_type="meanpool",
                                       model_size=self.model_size,
                                       identity_dim=self.identity_dim,
                                       logging=True)
        else:
            raise Exception('Error: model name unrecognized.')
        return model

    def train(self, train_data, sampler_name='Uniform'):
        print("Training model...")
        timer = Timer()
        timer.tic()

        G = train_data[0]
        features = train_data[1]
        id_map = train_data[2]

        if features is not None:
            # pad with dummy zero vector
            features = np.vstack([features, np.zeros((features.shape[1],))])

        context_pairs = train_data[3] if self.random_context else None
        placeholders = self._construct_placeholders()
        minibatch = EdgeMinibatchIterator(
                    G,
                    id_map,
                    placeholders,
                    batch_size=self.batch_size,
                    max_degree=self.max_degree,
                    num_neg_samples=self.neg_sample_size,
                    context_pairs=context_pairs)

        adj_info_ph = tf.compat.v1.placeholder(tf.int32,
                                               shape=minibatch.adj.shape)
        adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
        adj_shape = adj_info.get_shape().as_list()

        model = self._create_model(sampler_name, placeholders, features,
                                   adj_info, minibatch)

        config = tf.compat.v1.ConfigProto(
                log_device_placement=self.log_device_placement)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        # Initialize session
        sess = tf.compat.v1.Session(config=config)
        merged = tf.compat.v1.summary.merge_all()
#        summary_writer = tf.compat.v1.summary.FileWriter(
#                self._log_dir(sampler_name), sess.graph)

        # Initialize model saver
        saver = tf.compat.v1.train.Saver(max_to_keep=self.epochs)

        # Init variables
        sess.run(tf.compat.v1.global_variables_initializer(),
                 feed_dict={adj_info_ph: minibatch.adj})

        # Restore params of ML sampler model
        if sampler_name == 'ML' or sampler_name == 'FastML':
            sampler_vars = tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="MLsampler")
            saver_sampler = tf.compat.v1.train.Saver(var_list=sampler_vars)
            sampler_model_path = self._sampler_model_path()
            saver_sampler.restore(sess, sampler_model_path + 'model.ckpt')

        # Loss node path
        loss_node_path = self._loss_node_path(sampler_name)
        if not os.path.exists(loss_node_path):
            os.makedirs(loss_node_path)

        # Train model
        train_shadow_mrr = None
        shadow_mrr = None

        total_steps = 0
        avg_time = 0.0
        epoch_val_costs = []

        train_adj_info = tf.compat.v1.assign(adj_info, minibatch.adj)
        val_adj_info = tf.compat.v1.assign(adj_info, minibatch.test_adj)

        train_losses = []
        validation_losses = []

        val_cost_ = []
        val_mrr_ = []
        shadow_mrr_ = []
        duration_ = []

        ln_acc = sparse.csr_matrix((adj_shape[0], adj_shape[0]),
                                   dtype=np.float32)
        lnc_acc = sparse.csr_matrix((adj_shape[0], adj_shape[0]),
                                    dtype=np.int32)
        ln_acc = ln_acc.tolil()
        lnc_acc = lnc_acc.tolil()

        for epoch in range(self.epochs):
            minibatch.shuffle()

            iter = 0
            print('Epoch: %04d' % (epoch))
            epoch_val_costs.append(0)
            train_loss_epoch = []
            validation_loss_epoch = []

            while not minibatch.end():
                # Construct feed dictionary
                feed_dict = minibatch.next_minibatch_feed_dict()
                feed_dict.update({placeholders['dropout']: self.dropout})
                t = time.time()

                # Training step
                outs = sess.run([merged, model.opt_op, model.loss, model.ranks,
                                 model.aff_all, model.mrr, model.outputs1,
                                 model.loss_node, model.loss_node_count],
                                feed_dict=feed_dict)
                train_cost = outs[2]
                train_mrr = outs[5]
                train_loss_epoch.append(train_cost)

                if train_shadow_mrr is None:
                    train_shadow_mrr = train_mrr
                else:
                    train_shadow_mrr -= (1-0.99) * (
                            train_shadow_mrr - train_mrr)

                if iter % self.validate_iter == 0:
                    # Validation
                    sess.run(val_adj_info.op)
                    val_cost, ranks, val_mrr, duration  = self._evaluate(
                            sess, model, minibatch,
                            size=self.validate_batch_size)
                    sess.run(train_adj_info.op)
                    epoch_val_costs[-1] += val_cost
                    validation_loss_epoch.append(val_cost)

                if shadow_mrr is None:
                    shadow_mrr = val_mrr
                else:
                    shadow_mrr -= (1-0.99) * (shadow_mrr - val_mrr)

                val_cost_.append(val_cost)
                val_mrr_.append(val_mrr)
                shadow_mrr_.append(shadow_mrr)
                duration_.append(duration)

#                if total_steps % self.print_every == 0:
#                    summary_writer.add_summary(outs[0], total_steps)

                # Print results
                avg_time = (avg_time * total_steps + time.time() - t) / (
                        total_steps + 1)

                if total_steps % self.print_every == 0:
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

                ln = outs[7].values
                ln_idx = outs[7].indices
                ln_acc[ln_idx[:, 0], ln_idx[:, 1]] += ln

                lnc = outs[8].values
                lnc_idx = outs[8].indices
                lnc_acc[lnc_idx[:, 0], lnc_idx[:, 1]] += lnc

                iter += 1
                total_steps += 1

                if total_steps > self.max_total_steps:
                    break

            # Keep track of train and validation losses per epoch
            train_losses.append(sum(train_loss_epoch)/len(train_loss_epoch))
            validation_losses.append(
                    sum(validation_loss_epoch)/len(validation_loss_epoch))

            # If the epoch has the lowest validation loss so far
            if validation_losses[-1] == min(validation_losses):
                print("Minimum validation loss so far ({}) at epoch {}.".format(
                        validation_losses[-1], epoch))
                # Save loss node and count
                loss_node = sparse.save_npz(loss_node_path + 'loss_node.npz',
                                            sparse.csr_matrix(ln_acc))
                loss_node_count = sparse.save_npz(loss_node_path +
                                                  'loss_node_count.npz',
                                                  sparse.csr_matrix(lnc_acc))
                # Save embeddings
                if self.save_embeddings and sampler_name is not "Uniform":
                    sess.run(val_adj_info.op)
                    self._save_embeddings(sess, model, minibatch,
                                          self.validate_batch_size,
                                          self._log_dir(sampler_name))

            # Save model at each epoch
            print("Saving model at epoch {}.".format(epoch))
            saver.save(sess, os.path.join(
                       self._log_dir(sampler_name),
                       "model_epoch_" + str(epoch) + ".ckpt"))

            if total_steps > self.max_total_steps:
                break

        print("Optimization Finished!")

        training_time = timer.toc()
        self._plot_losses(train_losses, validation_losses, sampler_name)
        self._print_stats(train_losses, validation_losses, training_time,
                          sampler_name)

    def train_sampler(self, train_data):
        features = train_data[1]
        batch_size = 512

        if features is not None:
            features = np.vstack([features, np.zeros((features.shape[1],))])

        node_size = len(features)
        node_dim = len(features[0])

        # build model
        # input (features of vertex and its neighbor, label)
        x1_ph = tf.compat.v1.placeholder(shape=[batch_size, node_dim],
                                         dtype=tf.float32)
        x2_ph = tf.compat.v1.placeholder(shape=[batch_size, node_dim],
                                         dtype=tf.float32)
        y_ph = tf.compat.v1.placeholder(shape=[batch_size], dtype=tf.float32)

        with tf.compat.v1.variable_scope("MLsampler"):
            if self.nonlinear_sampler is True:
                print("Non-linear regression sampler used")
                l = tf.compat.v1.layers.dense(
                        tf.concat([x1_ph, x2_ph], axis=1), 1, activation=None,
                        trainable=True,
                        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                scale=1.0, mode="fan_avg",
                                distribution="uniform"),
                        name='dense')
                out = tf.nn.relu(tf.exp(l), name='relu')
            else:
                print("Linear regression sampler used")
                l = tf.compat.v1.layers.dense(
                        x1_ph, node_dim, activation=None, trainable=True,
                        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                scale=1.0, mode="fan_avg",
                                distribution="uniform"),
                        name='dense')
                l = tf.matmul(l, x2_ph, transpose_b=True, name='matmul')
                out = tf.nn.relu(l, name='relu')

        loss = tf.nn.l2_loss(out-y_ph, name='loss')/batch_size
        optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.learning_rate, name='Adam').minimize(loss)
        init = tf.compat.v1.global_variables_initializer()

        # configuration
        config = tf.compat.v1.ConfigProto(
                log_device_placement=self.log_device_placement)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        # load data
        loss_node_path = self._loss_node_path("Uniform")
        loss_node = sparse.load_npz(loss_node_path + 'loss_node.npz')
        loss_node_count = sparse.load_npz(loss_node_path +
                                          'loss_node_count.npz')

        idx_nz = sparse.find(loss_node_count)

        # due to out of memory, select randomly limited number of data node
        vertex = features[idx_nz[0]]
        neighbor = features[idx_nz[1]]
        count = idx_nz[2]
        y = np.divide(sparse.find(loss_node)[2],count)

        # partition train/validation data
        vertex_tr = vertex[:-batch_size]
        neighbor_tr = neighbor[:-batch_size]
        y_tr = y[:-batch_size]

        vertex_val = vertex[-batch_size:]
        neighbor_val = neighbor[-batch_size:]
        y_val = y[-batch_size:]

        iter_size = int(vertex_tr.shape[0]/batch_size)

        # initialize session
        sess = tf.compat.v1.Session(config=config)

        # summary
        tf.compat.v1.summary.scalar('loss', loss)
        merged_summary_op = tf.compat.v1.summary.merge_all()
#        summary_writer = tf.compat.v1.summary.FileWriter(
#                self._sampler_log_dir(), sess.graph)

        # save model
        saver = tf.compat.v1.train.Saver()
        model_path = self._sampler_model_path()
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # init variables
        sess.run(init)

        # train
        total_steps = 0
        avg_time = 0.0
        validation_losses = []

        for epoch in range(self.epochs):
            # shuffle
            perm = np.random.permutation(vertex_tr.shape[0])
            validation_loss_epoch = []

            print("Epoch: %04d" % (epoch))
            for iter in range(iter_size):
                # allocate batch
                vtr = vertex_tr[perm[iter*batch_size:(iter+1)*batch_size]]
                ntr = neighbor_tr[perm[iter*batch_size:(iter+1)*batch_size]]
                ytr = y_tr[perm[iter*batch_size:(iter+1)*batch_size]]

                t = time.time()
                outs = sess.run([loss, optimizer, merged_summary_op],
                                feed_dict={x1_ph: vtr, x2_ph: ntr, y_ph: ytr})
                train_loss = outs[0]

                # validation
                if iter % self.validate_iter == 0:
                    outs = sess.run([loss, optimizer, merged_summary_op],
                                    feed_dict={x1_ph: vertex_val,
                                               x2_ph: neighbor_val,
                                               y_ph: y_val})
                    val_loss = outs[0]
                    validation_loss_epoch.append(val_loss)

                avg_time = (avg_time*total_steps+time.time() - t)/(
                        total_steps+1)

                # print
                if total_steps % self.print_every == 0:
                    print("Iter:", "%04d" % iter,
                          "train_loss=", "{:.5f}".format(train_loss),
                          "val_loss=", "{:.5f}".format(val_loss))
                total_steps += 1

                if total_steps > self.max_total_steps:
                    break

            validation_losses.append(
                    sum(validation_loss_epoch)/len(validation_loss_epoch))
            if validation_losses[-1] == min(validation_losses):
                print("Minimum validation loss so far ({}) at epoch {}.".format(
                        validation_losses[-1], epoch))
                # save_model
                save_path = saver.save(sess, model_path+'model.ckpt')

        sess.close()
        tf.compat.v1.reset_default_graph()

    def predict(self, test_data, model_checkpoint, sampler_name="FastML",
                gpu_mem_fraction=None):
        timer = Timer()
        timer.tic()

        G = test_data[0]
        features = test_data[1]
        id_map = test_data[2]

        if features is not None:
            # pad with dummy zero vector
            features = np.vstack([features, np.zeros((features.shape[1],))])

        context_pairs = test_data[3] if self.random_context else None
        placeholders = self._construct_placeholders()
        minibatch = EdgeMinibatchIterator(
                    G,
                    id_map,
                    placeholders,
                    batch_size=self.batch_size,
                    max_degree=self.max_degree,
                    num_neg_samples=self.neg_sample_size,
                    context_pairs=context_pairs)

        adj_info_ph = tf.compat.v1.placeholder(tf.int32,
                                               shape=minibatch.adj.shape)
        adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
        adj_shape = adj_info.get_shape().as_list()

        model = self._create_model(sampler_name, placeholders, features,
                                   adj_info, minibatch)

        config = tf.compat.v1.ConfigProto(
                log_device_placement=self.log_device_placement)
        if gpu_mem_fraction is not None:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_fraction
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        # Initialize session
        sess = tf.compat.v1.Session(config=config)
        merged = tf.compat.v1.summary.merge_all()

        # Initialize model saver
        saver = tf.compat.v1.train.Saver()

        # Init variables
        sess.run(tf.compat.v1.global_variables_initializer(),
                 feed_dict={adj_info_ph: minibatch.adj})

        # Restore params of ML sampler model
        if sampler_name == 'ML' or sampler_name == 'FastML':
            sampler_vars = tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="MLsampler")
            saver_sampler = tf.compat.v1.train.Saver(var_list=sampler_vars)
            sampler_model_path = self._sampler_model_path()
            saver_sampler.restore(sess, sampler_model_path + 'model.ckpt')

        # Restore model
        print("Restoring trained model.")
        checkpoint_file = os.path.join(self._log_dir(sampler_name),
                                       model_checkpoint)
        ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_file)
        if checkpoint_file:
            saver.restore(sess, checkpoint_file)
            print("Model restored.")
        else:
            print("This model checkpoint does not exist. The model might " +
                  "not be trained yet or the checkpoint is invalid.")

        total_steps = 0
        avg_time = 0.0

        val_adj_info = tf.compat.v1.assign(adj_info, minibatch.test_adj)

        # Infer embeddings
        sess.run(val_adj_info.op)

        print("Computing embeddings...")
        val_embeddings = []
        finished = False
        seen = set([])
        nodes = []
        iter_num = 0
        while not finished:
            feed_dict_val, finished, edges = minibatch.incremental_embed_feed_dict(
                                            self.validate_batch_size, iter_num)
            iter_num += 1
            outs_val = sess.run([model.loss, model.mrr, model.outputs1],
                                feed_dict=feed_dict_val)
            for i, edge in enumerate(edges):
                if not edge[0] in seen:
                    val_embeddings.append(outs_val[-1][i, :])
                    nodes.append(edge[0])
                    seen.add(edge[0])
        val_embeddings = np.vstack(val_embeddings)
        if self.save_embeddings:
            print("Saving embeddings...")
            if not os.path.exists(self._log_dir(sampler_name)):
                os.makedirs(self._log_dir(sampler_name))
            np.save(self._log_dir(sampler_name) + "inferred_embeddings.npy",
                    val_embeddings)
            with open(self._log_dir(sampler_name) +
                      "inferred_embeddings_ids.txt", "w") as fp:
                fp.write("\n".join(map(str, nodes)))
            print("Embeddings saved.\n")

        # Return only the embeddings of the test nodes
        test_embeddings_ids = {}
        for i, node in enumerate(nodes):
            test_embeddings_ids[node] = i
        test_nodes = [n for n in G.nodes() if G.node[n]['test']]
        test_embeddings = val_embeddings[[test_embeddings_ids[id] for id in
                                          test_nodes]]

        sess.close()
        tf.compat.v1.reset_default_graph()
        timer.toc()
        return test_nodes, test_embeddings

    def _sampler_model_path(self):
        sampler_model_path = self.base_log_dir + self.train_prefix.rsplit(
                    "/", maxsplit=1)[-2] + "/MLsampler/unsupervised"
        sampler_model_path += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
                model=self.model_name, model_size=self.model_size,
                lr=self.learning_rate)
        return sampler_model_path

    def _loss_node_path(self, sampler_name):
        loss_node_path = self.base_log_dir + self.train_prefix.rsplit(
                    "/", maxsplit=1)[-2] + "/loss_node-" + sampler_name + \
                    "/unsupervised"
        loss_node_path += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
                model=self.model_name, model_size=self.model_size,
                lr=self.learning_rate)
        return loss_node_path

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for unsupervised GraphSAGE model.')
        parser.add_argument('train_prefix',
                            help='Name of the object file that stores the '
                            + 'training data.')
        parser.add_argument("model_name",
                            choices=["mean_concat", "mean_add", "gcn",
                                     "graphsage_seq", "graphsage_maxpool",
                                     "graphsage_meanpool"
                                     ],
                            help="Model names.")
        parser.add_argument("--nonlinear_sampler",
                            action="store_true",
                            default=False,
                            help="Where to use nonlinear sampler o.w. " +
                            "linear sampler"
                            )
        parser.add_argument("--uniform_ratio",
                            type=float,
                            default=0.6,
                            help="In case of FastML sampling, the " +
                            "percentile of uniform sampling preceding the " +
                            "regressor sampling")
        parser.add_argument('--model_size',
                            choices=["small", "big"],
                            default="small",
                            help="Can be big or small; model specific def'ns")
        parser.add_argument('--learning_rate',
                            type=float,
                            default=0.00001,
                            help='Initial learning rate.')
        parser.add_argument('--epochs',
                            type=int,
                            default=10,
                            help='Number of epochs to train.')
        parser.add_argument('--dropout',
                            type=float,
                            default=0.0,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--weight_decay',
                            type=float,
                            default=0.0,
                            help='Weight for l2 loss on embedding matrix.')
        parser.add_argument('--max_degree',
                            type=int,
                            default=100,
                            help='Maximum node degree.')
        parser.add_argument('--samples_1',
                            type=int,
                            default=25,
                            help='Number of samples in layer 1.')
        parser.add_argument('--samples_2',
                            type=int,
                            default=10,
                            help='Number of users samples in layer 2.')
        parser.add_argument('--dim_1',
                            type=int,
                            default=128,
                            help='Size of output dim ' +
                            '(final is 2x this, if using concat)')
        parser.add_argument('--dim_2',
                            type=int,
                            default=128,
                            help='Size of output dim ' +
                            '(final is 2x this, if using concat)')
        parser.add_argument('--random_context',
                            action="store_false",
                            default=True,
                            help='Whether to use random context or direct ' +
                            'edges.')
        parser.add_argument('--neg_sample_size',
                            type=int,
                            default=20,
                            help='Number of negative samples.')
        parser.add_argument('--batch_size',
                            type=int,
                            default=512,
                            help='Minibatch size.')
        parser.add_argument('--identity_dim',
                            type=int,
                            default=0,
                            help='Set to positive value to use identity ' +
                            'embedding features of that dimension.')
        parser.add_argument('--save_embeddings',
                            action="store_false",
                            default=True,
                            help='Whether to save embeddings for all nodes ' +
                            'after training')
        parser.add_argument('--base_log_dir',
                            default='../../../data/processed/graphsage_rl/',
                            help='Base directory for logging and saving ' +
                            'embeddings')
        parser.add_argument('--validate_iter',
                            type=int,
                            default=5000,
                            help='How often to run a validation minibatch.')
        parser.add_argument('--validate_batch_size',
                            type=int,
                            default=512,
                            help='How many nodes per validation sample.')
        parser.add_argument('--gpu',
                            type=int,
                            default=0,
                            help='Which gpu to use.')
        parser.add_argument('--print_every',
                            type=int,
                            default=50,
                            help='How often to print training info.')
        parser.add_argument('--max_total_steps',
                            type=int,
                            default=10**10,
                            help='Maximum total number of iterations.')
        parser.add_argument('--log_device_placement',
                            action="store_true",
                            default=False,
                            help='Whether to log device placement.')
        args = parser.parse_args()

        print("Starting...")
        print("Loading training data..")
        train_data = load_data(args.train_prefix, load_walks=True)
        print("Done loading training data..\n")

        from unsupervised_model import UnsupervisedModelRL
        model = UnsupervisedModelRL(args.train_prefix, args.model_name,
                                    args.nonlinear_sampler, args.uniform_ratio,
                                    args.model_size, args.learning_rate,
                                    args.epochs, args.dropout,
                                    args.weight_decay, args.max_degree,
                                    args.samples_1, args.samples_2, args.dim_1,
                                    args.dim_2, args.random_context,
                                    args.neg_sample_size, args.batch_size,
                                    args.identity_dim, args.save_embeddings,
                                    args.base_log_dir, args.validate_iter,
                                    args.validate_batch_size, args.gpu,
                                    args.print_every, args.max_total_steps,
                                    args.log_device_placement)

        print("Start training uniform sampling + graphsage model..")
        p_train_uniform = mp.Process(target=model.train,
                                     args=(train_data, "Uniform"))
        p_train_uniform.start()
        p_train_uniform.join()
        p_train_uniform.terminate()
        print("Done training uniform sampling + graphsage model..")

        print("Start training ML sampler..")
        p_train_sampler = mp.Process(target=model.train_sampler,
                                     args=(train_data, ))
        p_train_sampler.start()
        p_train_sampler.join()
        p_train_sampler.terminate()
        print("Done training ML sampler..")

        print("Start training ML sampling + graphsage model..")
        p_train_ml = mp.Process(target=model.train,
                                args=(train_data, "FastML"))
        p_train_ml.start()
        p_train_ml.join()
        p_train_ml.terminate()
        print("Done training ML sampling + graphsage model..")

        print("Finished.")

    if __name__ == "__main__":
        main()
