# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import pdb
import argparse
import numpy as np
from scipy import sparse
import sklearn
from sklearn import metrics
import multiprocessing as mp
import matplotlib.pyplot as plt
import tensorflow as tf

from supervised_models import SupervisedGraphsage
from models import SAGEInfo
from minibatch import NodeMinibatchIterator
from neigh_samplers import UniformNeighborSampler
from neigh_samplers import MLNeighborSampler, FastMLNeighborSampler
from utils import load_data

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))
from TimerCounter import Timer

# DISCLAIMER:
# This code file is derived from https://github.com/oj9040/GraphSAGE_RL.


class SupervisedModelRL:

    def __init__(self, train_prefix, model_name, nonlinear_sampler=True,
                 fast_ver=False, allhop_rewards=False, model_size="small",
                 learning_rate=0.001, epochs=10, dropout=0.0, weight_decay=0.0,
                 max_degree=100, samples_1=25, samples_2=10, samples_3=0,
                 dim_1=512, dim_2=512, dim_3=0, batch_size=128, sigmoid=False,
                 identity_dim=0,
                 base_log_dir='../../../data/processed/graphsage_rl/',
                 validate_iter=5000, validate_batch_size=128, gpu=0,
                 print_every=5, max_total_steps=10**10,
                 log_device_placement=False):

        self.train_prefix = train_prefix
        self.model_name = model_name
        self.nonlinear_sampler = nonlinear_sampler
        self.fast_ver = fast_ver
        self.allhop_rewards = allhop_rewards
        self.model_size = model_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.max_degree = max_degree
        self.samples_1 = samples_1
        self.samples_2 = samples_2
        self.samples_3 = samples_3
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.dim_3 = dim_3
        self.batch_size = batch_size
        self.sigmoid = sigmoid
        self.identity_dim = identity_dim
        self.base_log_dir = base_log_dir
        if base_log_dir == "../../../data/processed/graphsage_rl/":
            self.base_log_dir = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "..", "..", "..", "data", "processed", "graphsage_rl")
        else:
            self.base_log_dir = base_log_dir
        self.validate_iter = validate_iter
        self.validate_batch_size = validate_batch_size
        self.gpu = gpu
        self.print_every = print_every
        self.max_total_steps = max_total_steps
        self.log_device_placement = log_device_placement

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Set random seed
        seed = 123
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        tf.compat.v1.disable_eager_execution()
        print("GraphSAGERL Model intialized.")

    def _model_prefix(self):
        model_prefix = 'f' + str(self.dim_1) + '_' + str(self.dim_2) + '_' + \
                        str(self.dim_3) + '-s' + str(self.samples_1) + '_' + \
                        str(self.samples_2) + '_' + str(self.samples_3)
        return model_prefix

    def _hyper_prefix(self):
        hyper_prefix = "/{model:s}-{model_size:s}-lr{lr:0.4f}-bs{batch_size:d}-ep{epochs:d}/".format(
                model=self.model_name, model_size=self.model_size,
                lr=self.learning_rate, batch_size=self.batch_size,
                epochs=self.epochs)
        return hyper_prefix

    def _log_dir(self, sampler_model_name):
        log_dir = self.base_log_dir + "/" \
                    + self.train_prefix.rsplit("/", maxsplit=1)[-2] + "-" \
                    + self._model_prefix() + "-" + sampler_model_name
        log_dir += self._hyper_prefix()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def _sampler_log_dir(self):
        log_dir = self.base_log_dir + "/" + \
                  self.train_prefix.rsplit("/", maxsplit=1)[-2] + \
                  '-' + self._model_prefix()
        log_dir += self._hyper_prefix()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    # Calculate only micro f1 score
    def _calc_f1(self, y_true, y_pred):
        if not self.sigmoid:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0
        f1_micro = metrics.f1_score(y_true, y_pred, average="micro")
        return f1_micro, f1_micro

    # Define model evaluation function
    def _evaluate(self, sess, model, minibatch_iter, size=None):
        t_test = time.time()
        feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict_val)
        mic, mac = self._calc_f1(labels, node_outs_val[0])
        return node_outs_val[1], mic, mac, (time.time() - t_test)

    def _incremental_evaluate(self, sess, model, minibatch_iter, size,
                              run_options=None, run_metadata=None, test=False):
        t_test = time.time()
        finished = False
        val_losses = []
        val_preds = []
        labels = []
        iter_num = 0
        while not finished:
            feed_dict_val, batch_labels, finished, _  = minibatch_iter.incremental_node_val_feed_dict(
                    size, iter_num, test=test)
            node_outs_val = sess.run([model.preds, model.loss],
                                     feed_dict=feed_dict_val,
                                     options=run_options,
                                     run_metadata=run_metadata)
            val_preds.append(node_outs_val[0])
            labels.append(batch_labels)
            val_losses.append(node_outs_val[1])
            iter_num += 1
        val_preds = np.vstack(val_preds)
        labels = np.vstack(labels)
        f1_scores = self._calc_f1(labels, val_preds)
        return np.mean(val_losses), f1_scores[0], f1_scores[1], (
                time.time() - t_test)

    def _construct_placeholders(self, num_classes):
        # Define placeholders
        placeholders = {
            'labels': tf.compat.v1.placeholder(
                    tf.float32, shape=(None, num_classes), name='labels'),
            'batch': tf.compat.v1.placeholder(
                    tf.int32, shape=(self.batch_size), name='batch1'),
            'dropout': tf.compat.v1.placeholder_with_default(
                    0., shape=(), name='dropout'),
            'batch_size': tf.compat.v1.placeholder(
                    tf.int32, name='batch_size'),
            'learning_rate': tf.compat.v1.placeholder(
                    tf.float32, name='learning_rate')
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

    def _create_model(self, sampler_name, num_classes, placeholders, features,
                      adj_info, minibatch):
        if self.model_name == 'mean_concat':
            # Create model
            sampler = self._create_sampler(sampler_name, adj_info, features)
            if self.samples_3 != 0:
                layer_infos = [
                        SAGEInfo("node", sampler, self.samples_1, self.dim_1),
                        SAGEInfo("node", sampler, self.samples_2, self.dim_2),
                        SAGEInfo("node", sampler, self.samples_3, self.dim_3)]
            elif self.samples_2 != 0:
                layer_infos = [
                        SAGEInfo("node", sampler, self.samples_1, self.dim_1),
                        SAGEInfo("node", sampler, self.samples_2, self.dim_2)]
            else:
                layer_infos = [
                        SAGEInfo("node", sampler, self.samples_1, self.dim_1)]

            # modified
            model = SupervisedGraphsage(num_classes,
                                        placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos,
                                        weight_decay=self.weight_decay,
                                        learning_rate=self.learning_rate,
                                        batch_size=self.batch_size,
                                        concat=True,
                                        model_size=self.model_size,
                                        sigmoid_loss=self.sigmoid,
                                        identity_dim=self.identity_dim,
                                        logging=True)

        elif self.model_name == 'mean_add':
            # Create model
            sampler = self._create_sampler(sampler_name, adj_info, features)
            if self.samples_3 != 0:
                layer_infos = [
                        SAGEInfo("node", sampler, self.samples_1, self.dim_1),
                        SAGEInfo("node", sampler, self.samples_2, self.dim_2),
                        SAGEInfo("node", sampler, self.samples_3, self.dim_3)]
            elif self.samples_2 != 0:
                layer_infos = [
                        SAGEInfo("node", sampler, self.samples_1, self.dim_1),
                        SAGEInfo("node", sampler, self.samples_2, self.dim_2)]
            else:
                layer_infos = [
                        SAGEInfo("node", sampler, self.samples_1, self.dim_1)]

            # modified
            model = SupervisedGraphsage(num_classes,
                                        placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos,
                                        weight_decay=self.weight_decay,
                                        learning_rate=self.learning_rate,
                                        batch_size=self.batch_size,
                                        concat=False,
                                        model_size=self.model_size,
                                        sigmoid_loss=self.sigmoid,
                                        identity_dim=self.identity_dim,
                                        logging=True)
        elif self.model_name == 'gcn':
            # Create model
            sampler = self._create_sampler(sampler_name, adj_info, features)
            layer_infos = [
                    SAGEInfo("node", sampler, self.samples_1, 2*self.dim_1),
                    SAGEInfo("node", sampler, self.samples_2, 2*self.dim_2)]
            model = SupervisedGraphsage(num_classes,
                                        placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos=layer_infos,
                                        weight_decay=self.weight_decay,
                                        learning_rate=self.learning_rate,
                                        batch_size=self.batch_size,
                                        aggregator_type="gcn",
                                        model_size=self.model_size,
                                        concat=False,
                                        sigmoid_loss=self.sigmoid,
                                        identity_dim=self.identity_dim,
                                        logging=True)
        elif self.model_name == 'graphsage_seq':
            # Create model
            sampler = self._create_sampler("Uniform", adj_info, features)
            layer_infos = [
                    SAGEInfo("node", sampler, self.samples_1, self.dim_1),
                    SAGEInfo("node", sampler, self.samples_2, self.dim_2)]
            model = SupervisedGraphsage(num_classes,
                                        placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos=layer_infos,
                                        weight_decay=self.weight_decay,
                                        learning_rate=self.learning_rate,
                                        batch_size=self.batch_size,
                                        aggregator_type="seq",
                                        model_size=self.model_size,
                                        sigmoid_loss=self.sigmoid,
                                        identity_dim=self.identity_dim,
                                        logging=True)
        elif self.model_name == 'graphsage_maxpool':
            # Create model
            sampler = self._create_sampler(sampler_name, adj_info, features)
            layer_infos = [
                    SAGEInfo("node", sampler, self.samples_1, self.dim_1),
                    SAGEInfo("node", sampler, self.samples_2, self.dim_2)]
            model = SupervisedGraphsage(num_classes,
                                        placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos=layer_infos,
                                        weight_decay=self.weight_decay,
                                        learning_rate=self.learning_rate,
                                        batch_size=self.batch_size,
                                        aggregator_type="maxpool",
                                        model_size=self.model_size,
                                        sigmoid_loss=self.sigmoid,
                                        identity_dim=self.identity_dim,
                                        logging=True)
        elif self.model_name == 'graphsage_meanpool':
            # Create model
            sampler = self._create_sampler(sampler_name, adj_info, features)
            layer_infos = [
                    SAGEInfo("node", sampler, self.samples_1, self.dim_1),
                    SAGEInfo("node", sampler, self.samples_2, self.dim_2)]
            model = SupervisedGraphsage(num_classes,
                                        placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos=layer_infos,
                                        weight_decay=self.weight_decay,
                                        learning_rate=self.learning_rate,
                                        batch_size=self.batch_size,
                                        aggregator_type="meanpool",
                                        model_size=self.model_size,
                                        sigmoid_loss=self.sigmoid,
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
        class_map = train_data[4]

        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
        else:
            num_classes = len(set(class_map.values()))

        if features is not None:
            # pad with dummy zero vector
            features = np.vstack([features, np.zeros((features.shape[1],))])

        placeholders = self._construct_placeholders(num_classes)
        minibatch = NodeMinibatchIterator(
                G,
                id_map,
                placeholders,
                class_map,
                num_classes,
                batch_size=self.batch_size,
                max_degree=self.max_degree)
        adj_info_ph = tf.compat.v1.placeholder(tf.int32,
                                               shape=minibatch.adj.shape)
        adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
        adj_shape = adj_info.get_shape().as_list()

        model = self._create_model(sampler_name, num_classes, placeholders,
                                   features, adj_info, minibatch)

        config = tf.compat.v1.ConfigProto(
                log_device_placement=self.log_device_placement)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        # Initialize session
        sess = tf.compat.v1.Session(config=config)
        merged = tf.compat.v1.summary.merge_all()
#        summary_writer = tf.compat.v1.summary.FileWriter(
#                self._log_dir(sampler_name), sess.graph)

        # Save model
        saver = tf.compat.v1.train.Saver()
        loss_node_path = self._loss_node_path(sampler_name)

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

        # Train model
        total_steps = 0
        avg_time = 0.0
        epoch_val_costs = []
        train_losses = []
        validation_losses = []

        train_adj_info = tf.compat.v1.assign(adj_info, minibatch.adj)
        val_adj_info = tf.compat.v1.assign(adj_info, minibatch.test_adj)

        val_cost_ = []
        val_f1_mic_ = []
        val_f1_mac_ = []
        duration_ = []
        epoch_laps_ = []

        ln_acc = sparse.csr_matrix((adj_shape[0], adj_shape[0]),
                                   dtype=np.float32)
        lnc_acc = sparse.csr_matrix((adj_shape[0], adj_shape[0]),
                                    dtype=np.int32)
        ln_acc = ln_acc.tolil()
        lnc_acc = lnc_acc.tolil()

        learning_rate = [self.learning_rate]

        for lr_iter in range(len(learning_rate)):
            for epoch in range(self.epochs):
                epoch_time = time.time()
                minibatch.shuffle()

                iter = 0
                print('Epoch: %04d' % (epoch))
                epoch_val_costs.append(0)
                train_loss_epoch = []
                validation_loss_epoch = []

                while not minibatch.end():
                    # Construct feed dictionary
                    feed_dict, labels = minibatch.next_minibatch_feed_dict()
                    feed_dict.update({placeholders['dropout']: self.dropout})
                    feed_dict.update({
                            placeholders['learning_rate']: learning_rate[
                                    lr_iter]})
                    t = time.time()

                    # Training step
                    outs = sess.run([merged, model.opt_op, model.loss,
                                     model.preds, model.loss_node,
                                     model.loss_node_count],
                                    feed_dict=feed_dict)
                    train_cost = outs[2]
                    train_loss_epoch.append(train_cost)

                    if iter % self.validate_iter == 0:
                        # Validation
                        sess.run(val_adj_info.op)
                        if self.validate_batch_size == -1:
                            val_cost, val_f1_mic, val_f1_mac, duration = self._incremental_evaluate(
                                    sess, model, minibatch, self.batch_size)
                        else:
                            val_cost, val_f1_mic, val_f1_mac, duration = self._evaluate(
                                    sess, model, minibatch,
                                    self.validate_batch_size)

                        # accumulate val results
                        val_cost_.append(val_cost)
                        val_f1_mic_.append(val_f1_mic)
                        val_f1_mac_.append(val_f1_mac)
                        duration_.append(duration)

                        sess.run(train_adj_info.op)
                        epoch_val_costs[-1] += val_cost
                        validation_loss_epoch.append(val_cost)

#                    if total_steps % self.print_every == 0:
#                        summary_writer.add_summary(outs[0], total_steps)

                    # Print results
                    avg_time = (avg_time * total_steps + time.time() - t) / (
                            total_steps + 1)

                    ln = outs[4].values
                    ln_idx = outs[4].indices
                    ln_acc[ln_idx[:, 0], ln_idx[:, 1]] += ln

                    lnc = outs[5].values
                    lnc_idx = outs[5].indices
                    lnc_acc[lnc_idx[:, 0], lnc_idx[:, 1]] += lnc

                    if total_steps % self.print_every == 0:
                        train_f1_mic, train_f1_mac = self._calc_f1(
                                labels, outs[3])
                        print("Iter:", '%04d' % iter,
                              "train_loss=", "{:.5f}".format(train_cost),
                              "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                              "val_loss=", "{:.5f}".format(val_cost),
                              "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                              "time per iter=", "{:.5f}".format(avg_time))

                    iter += 1
                    total_steps += 1

                    if total_steps > self.max_total_steps:
                        break

                # Keep track of train and validation losses per epoch
                train_losses.append(sum(train_loss_epoch)/len(train_loss_epoch)
                                    )
                validation_losses.append(
                        sum(validation_loss_epoch)/len(validation_loss_epoch))

                # If the epoch has the lowest validation loss so far
                if validation_losses[-1] == min(validation_losses):
                    print("Minimum validation loss so far ({}) at epoch {}.".format(
                            validation_losses[-1], epoch))
                    # Save loss node and count
                    loss_node = sparse.save_npz(
                            loss_node_path + 'loss_node.npz',
                            sparse.csr_matrix(ln_acc))
                    loss_node_count = sparse.save_npz(
                            loss_node_path + 'loss_node_count.npz',
                            sparse.csr_matrix(lnc_acc))

                    print("Saving model at epoch {}.".format(epoch))
                    saver.save(sess, os.path.join(self._log_dir(sampler_name),
                                                  "model.ckpt"))

                epoch_laps = time.time() - epoch_time
                epoch_laps_.append(epoch_laps)
                print("Epoch time=", "{:.5f}".format(epoch_laps))

                if total_steps > self.max_total_steps:
                    break

        print("Avg time per epoch=", "{:.5f}".format(np.mean(epoch_laps_)))

        print("Optimization Finished!")
        training_time = timer.toc()
        self._plot_losses(train_losses, validation_losses, sampler_name)
        self._print_stats(train_losses, validation_losses, training_time,
                          sampler_name)

        sess.close()
        tf.compat.v1.reset_default_graph()

    def train_sampler(self, train_data, sampler_name="ML"):
        features = train_data[1]
        batch_size = 512

        if features is not None:
            features = np.vstack([features, np.zeros((features.shape[1],))])

        node_size = len(features)
        node_dim = len(features[0])

        # Build model
        # Input (features of vertex and its neighbor, label)
        x1_ph = tf.compat.v1.placeholder(shape=[batch_size, node_dim],
                                         dtype=tf.float32)
        x2_ph = tf.compat.v1.placeholder(shape=[batch_size, node_dim],
                                         dtype=tf.float32)
        y_ph = tf.compat.v1.placeholder(shape=[batch_size], dtype=tf.float32)
        lr_ph = tf.compat.v1.placeholder(dtype=tf.float32)

        # Sampler model (non-linear, linear)
        with tf.compat.v1.variable_scope("MLsampler"):
            if self.nonlinear_sampler is True:
                print("Non-linear regression sampler used.")
                l = tf.compat.v1.layers.dense(
                        tf.concat([x1_ph, x2_ph], axis=1), 1,
                        activation=tf.nn.relu, trainable=True,
                        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                scale=1.0, mode="fan_avg",
                                distribution="uniform"),
                        name='dense')
                out = tf.exp(l)
            else:
                print("Linear regression sampler used.")
                l = tf.compat.v1.layers.dense(
                        x1_ph, node_dim, activation=None, trainable=True,
                        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                scale=1.0, mode="fan_avg",
                                distribution="uniform"),
                        name='dense')
                l = tf.matmul(l, x2_ph, transpose_b=True, name='matmul')
                out = tf.nn.relu(l, name='relu')

        # l2 loss
        loss = tf.nn.l2_loss(out-y_ph, name='loss')/batch_size

        # Optimizer
        optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=lr_ph, name='Adam').minimize(loss)
        init = tf.compat.v1.global_variables_initializer()

        # Configuration
        config = tf.compat.v1.ConfigProto(
                log_device_placement=self.log_device_placement)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        # Construct reward from loss
        if self.allhop_rewards:
            # Using all-hop rewards
            dims = [self.dim_1, self.dim_2, self.dim_3]
            samples = [self.samples_1, self.samples_2, self.samples_3]
            numhop = np.count_nonzero(samples)
            gamma = 0.9
            loss_node = 0
            loss_node_count = 0
            for i in reversed(range(0, numhop)):
                model_prefix_ = 'f' + str(dims[0]) + '_' + str(dims[1]) + '_'\
                                + str(dims[2]) + '-s' + str(samples[0]) + '_' \
                                + str(samples[1]) + '_' + str(samples[2])

                # Load data
                loss_node_path = self.base_log_dir + self.train_prefix.rsplit(
                                 "/", maxsplit=1)[-2] + "/loss_node-" + \
                                 model_prefix_ + "-Uniform"
                loss_node_path += self._hyper_prefix()
                loss_node_perstep = sparse.load_npz(loss_node_path +
                                                    'loss_node.npz')
                loss_node_count_perstep = sparse.load_npz(loss_node_path +
                                                          'loss_node_count.npz'
                                                          )

                loss_node += (gamma**i)*loss_node_perstep
                loss_node_count += loss_node_count_perstep

                dims[i] = 0
                samples[i] = 0
        else:
            # Using only last-hop reward
            # Load data
            loss_node_path = self._loss_node_path("Uniform")
            loss_node = sparse.load_npz(loss_node_path + 'loss_node.npz')
            loss_node_count = sparse.load_npz(loss_node_path +
                                              'loss_node_count.npz')

        cnt_nz = sparse.find(loss_node_count)
        loss_nz = sparse.find(loss_node)

        # Subsampling if the number of loss nodes is very large
        if cnt_nz[0].shape[0] > 1000000:
            cnt_nz_samp = np.int32(np.random.uniform(0, cnt_nz[0].shape[0]-1,
                                                     1000000))
            cnt_nz_v = cnt_nz[0][cnt_nz_samp]
            cnt_nz_n = cnt_nz[1][cnt_nz_samp]
            cnt = cnt_nz[2][cnt_nz_samp]
            lss = loss_nz[2][cnt_nz_samp]
        else:
            cnt_nz_v = cnt_nz[0]
            cnt_nz_n = cnt_nz[1]
            cnt = cnt_nz[2]
            lss = loss_nz[2]

        vertex = features[cnt_nz_v]
        neighbor = features[cnt_nz_n]
        y = np.divide(lss, cnt)

        # Plot histogram of reward
        fig = plt.hist(y, bins=128, range=(0, np.mean(y)*2), alpha=0.7,
                       color='k')
        plt.xlabel('Value')
        plt.ylabel('Number')
        plt.savefig(loss_node_path + 'histogram_valuefunc.png')

        # Partition train/validation data
        vertex_tr = vertex[:-batch_size]
        neighbor_tr = neighbor[:-batch_size]
        y_tr = y[:-batch_size]

        vertex_val = vertex[-batch_size:]
        neighbor_val = neighbor[-batch_size:]
        y_val = y[-batch_size:]

        iter_size = int(vertex_tr.shape[0]/batch_size)

        # Initialize session
        sess = tf.compat.v1.Session(config=config)

        # Summary
        tf.compat.v1.summary.scalar('loss', loss)
        merged_summary_op = tf.compat.v1.summary.merge_all()
#        summary_writer = tf.compat.v1.summary.FileWriter(
#                self._sampler_log_dir(), sess.graph)

        # Save model
        model_vars = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        saver = tf.compat.v1.train.Saver(var_list=model_vars)

        model_path = self._sampler_model_path()

        # Init variables
        sess.run(init)

        # Train
        total_steps = 0
        avg_time = 0.0
        validation_losses = []

        # Learning rate of sampler needs to be smaller than gnn's for
        # stable optimization
        lr = [self.learning_rate/10]
        val_loss_old = 0

        for lr_iter in range(len(lr)):
            print('Learning rate= %f' % lr[lr_iter])
            for epoch in range(50):
                # shuffle
                perm = np.random.permutation(vertex_tr.shape[0])
                validation_loss_epoch = []
                print("Epoch: %04d" % (epoch))

                for iter in range(iter_size):
                    # allocate batch
                    vtr = vertex_tr[perm[iter*batch_size:(iter+1)*batch_size]]
                    ntr = neighbor_tr[perm[iter*batch_size:(iter+1)*batch_size]
                                      ]
                    ytr = y_tr[perm[iter*batch_size:(iter+1)*batch_size]]

                    t = time.time()
                    outs = sess.run([loss, optimizer, merged_summary_op],
                                    feed_dict={x1_ph: vtr, x2_ph: ntr,
                                               y_ph: ytr, lr_ph: lr[lr_iter]})
                    train_loss = outs[0]

                    # validation
                    if iter % self.validate_iter == 0:
                        outs = sess.run([loss, optimizer, merged_summary_op],
                                        feed_dict={x1_ph: vertex_val,
                                                   x2_ph: neighbor_val,
                                                   y_ph: y_val,
                                                   lr_ph: lr[lr_iter]})
                        val_loss = outs[0]
                        if val_loss == val_loss_old:
                            sess.close()
                            tf.compat.v1.reset_default_graph()
                            return 0
                        else:
                            val_loss_old = val_loss
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
                # If the epoch has the lowest validation loss so far
                if validation_losses[-1] == min(validation_losses):
                    print("Minimum validation loss so far ({}) at epoch {}.".format(
                            validation_losses[-1], epoch))
                    print("Saving model at epoch {}.".format(epoch))
                    saver.save(sess, model_path + 'model.ckpt')

        sess.close()
        tf.compat.v1.reset_default_graph()

    def inference(self, test_data, sampler_name):
        print("Inference...")
        tf.compat.v1.reset_default_graph()
        timer = Timer()
        timer.tic()

        G = test_data[0]
        features = test_data[1]
        id_map = test_data[2]
        class_map = test_data[4]

        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
        else:
            num_classes = len(set(class_map.values()))

        if features is not None:
            # pad with dummy zero vector
            features = np.vstack([features, np.zeros((features.shape[1],))])

        placeholders = self._construct_placeholders(num_classes)
        minibatch = NodeMinibatchIterator(
                G,
                id_map,
                placeholders,
                class_map,
                num_classes,
                batch_size=self.batch_size,
                max_degree=self.max_degree)
        adj_info_ph = tf.compat.v1.placeholder(tf.int32,
                                               shape=minibatch.adj.shape)
        adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

        model = self._create_model(sampler_name, num_classes, placeholders,
                                   features, adj_info, minibatch)

        config = tf.compat.v1.ConfigProto(
                log_device_placement=self.log_device_placement)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        # Initialize session
        sess = tf.compat.v1.Session(config=config)
        merged = tf.compat.v1.summary.merge_all()

        # Initialize model saver
        model_vars = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        saver = tf.compat.v1.train.Saver(var_list=model_vars)

        # Init variables
        sess.run(tf.compat.v1.global_variables_initializer(),
                 feed_dict={adj_info_ph: minibatch.adj})

        # Restore model
        print("Restoring trained model.")
        checkpoint_file = os.path.join(self._log_dir(sampler_name),
                                       "model.ckpt")
        print("Checkpoint file: {}".format(checkpoint_file))
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
        sess.run(val_adj_info.op)

        print("Computing predictions...")
        t_test = time.time()
        finished = False
        val_losses = []
        val_preds = []
        labels = []
        nodes = []
        iter_num = 0
        while not finished:
            feed_dict_val, batch_labels, finished, nodes_subset  = minibatch.incremental_node_val_feed_dict(
                    self.batch_size, iter_num, test=True)
            node_outs_val = sess.run([model.preds, model.loss],
                                     feed_dict=feed_dict_val)
            val_preds.append(node_outs_val[0])
            labels.append(batch_labels)
            val_losses.append(node_outs_val[1])
            nodes.extend(nodes_subset)
            iter_num += 1
        val_preds = np.vstack(val_preds)
        print("Computed.")

        # Return only the embeddings of the test nodes
        test_preds_ids = {}
        for i, node in enumerate(nodes):
            test_preds_ids[node] = i
        test_nodes = [n for n in G.nodes() if G.node[n]['test']]
        test_preds = val_preds[[test_preds_ids[id] for id in test_nodes]]
        timer.toc()
        sess.close()

        return test_nodes, test_preds

    def _sampler_model_path(self):
        if self.allhop_rewards:
            sampler_model_path = self.base_log_dir + self.train_prefix.rsplit(
                        "/", maxsplit=1)[-2] + "/MLsampler-" + \
                        self._model_prefix() + "-allhops"
        else:
            sampler_model_path = self.base_log_dir + self.train_prefix.rsplit(
                        "/", maxsplit=1)[-2] + "/MLsampler-" + \
                        self._model_prefix() + "-lasthop"
        sampler_model_path += self._hyper_prefix()
        if not os.path.exists(sampler_model_path):
            os.makedirs(sampler_model_path)
        return sampler_model_path

    def _loss_node_path(self, sampler_name):
        loss_node_path = self.base_log_dir + self.train_prefix.rsplit(
                         "/", maxsplit=1)[-2] + "/loss_node-" + \
                         self._model_prefix() + "-" + sampler_name
        loss_node_path += self._hyper_prefix()
        if not os.path.exists(loss_node_path):
            os.makedirs(loss_node_path)
        return loss_node_path

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for supervised GraphSAGE_RL model.')
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
                            action="store_false",
                            default=True,
                            help="Where to use nonlinear sampler o.w. " +
                            "linear sampler"
                            )
        parser.add_argument("--fast_ver",
                            action="store_true",
                            default=False,
                            help="Whether to use a fast version of the " +
                            "nonlinear sampler"
                            )
        parser.add_argument("--allhop_rewards",
                            action="store_true",
                            default=False,
                            help="Whether to use a all-hop rewards or " +
                            "last-hop reward for training the nonlinear " +
                            "sampler"
                            )
        parser.add_argument('--model_size',
                            choices=["small", "big"],
                            default="small",
                            help="Can be big or small; model specific def'ns")
        parser.add_argument('--learning_rate',
                            type=float,
                            default=0.001,
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
        parser.add_argument('--samples_3',
                            type=int,
                            default=0,
                            help='Number of users samples in layer 3. ' +
                            '(Only for mean model)')
        parser.add_argument('--dim_1',
                            type=int,
                            default=512,
                            help='Size of output dim ' +
                            '(final is 2x this, if using concat)')
        parser.add_argument('--dim_2',
                            type=int,
                            default=512,
                            help='Size of output dim ' +
                            '(final is 2x this, if using concat)')
        parser.add_argument('--dim_3',
                            type=int,
                            default=0,
                            help='Size of output dim ' +
                            '(final is 2x this, if using concat)')
        parser.add_argument('--batch_size',
                            type=int,
                            default=128,
                            help='Minibatch size.')
        parser.add_argument('--sigmoid',
                            action="store_true",
                            default=False,
                            help='Whether to use sigmoid loss ')
        parser.add_argument('--identity_dim',
                            type=int,
                            default=0,
                            help='Set to positive value to use identity ' +
                            'embedding features of that dimension.')
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
                            default=128,
                            help='How many nodes per validation sample.')
        parser.add_argument('--gpu',
                            type=int,
                            default=0,
                            help='Which gpu to use.')
        parser.add_argument('--print_every',
                            type=int,
                            default=5,
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
        train_data = load_data(args.train_prefix)
        print("Done loading training data..\n")

        from supervised_model import SupervisedModelRL
        model = SupervisedModelRL(args.train_prefix, args.model_name,
                                  args.nonlinear_sampler, args.fast_ver,
                                  args.allhop_rewards, args.model_size,
                                  args.learning_rate, args.epochs,
                                  args.dropout, args.weight_decay,
                                  args.max_degree, args.samples_1,
                                  args.samples_2, args.samples_3, args.dim_1,
                                  args.dim_2, args.dim_3, args.batch_size,
                                  args.sigmoid, args.identity_dim,
                                  args.base_log_dir, args.validate_iter,
                                  args.validate_batch_size, args.gpu,
                                  args.print_every, args.max_total_steps,
                                  args.log_device_placement)

        print("Start 1st phase: training graphsage model w/ " +
              "uniform sampling...")
        if model.allhop_rewards:
            dim_2_org = model.dim_2
            dim_3_org = model.dim_3
            samples_2_org = model.samples_2
            samples_3_org = model.samples_3

            dims = [model.dim_1, model.dim_2, model.dim_3]
            samples = [model.samples_1, model.samples_2, model.samples_3]
            numhop = np.count_nonzero(samples)
            for i in reversed(range(0, numhop)):
                model.dim_2 = dims[1]
                model.dim_3 = dims[2]
                model.samples_2 = samples[1]
                model.samples_3 = samples[2]
                print('Obtainining %d/%d hop reward' % (i+1, numhop))
                p_train_uniform = mp.Process(target=model.train,
                                             args=(train_data, "Uniform"))
                p_train_uniform.start()
                p_train_uniform.join()
                p_train_uniform.terminate()

                dims[i] = 0
                samples[i] = 0

            model.dim_2 = dim_2_org
            model.dim_3 = dim_3_org
            model.samples_2 = samples_2_org
            model.samples_3 = samples_3_org
        else:
            p_train_uniform = mp.Process(target=model.train,
                                         args=(train_data, "Uniform"))
            p_train_uniform.start()
            p_train_uniform.join()
            p_train_uniform.terminate()
        print("Done 1st phase: training graphsage model w/ uniform sampling..")

        # Train sampler
        print("Training RL-based regressor...")
        p_train_sampler = mp.Process(target=model.train_sampler,
                                     args=(train_data, ))
        p_train_sampler.start()
        p_train_sampler.join()
        p_train_sampler.terminate()
        print("Done training RL-based regressor...")

        # Train
        print("Start 2nd phase: training graphsage model w/ " +
              "data-driven sampler...")
        if args.fast_ver:
            p_train_ml = mp.Process(target=model.train,
                                    args=(train_data, "FastML"))
        else:
            p_train_ml = mp.Process(target=model.train,
                                    args=(train_data, "ML"))
        p_train_ml.start()
        p_train_ml.join()
        p_train_ml.terminate()
        print("Done 2nd phase: training graphsage model w/ data-driven " +
              "sampler...")
        print("Finished.")

    if __name__ == "__main__":
        main()
