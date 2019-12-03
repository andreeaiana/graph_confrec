# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import argparse
import pickle
import sklearn
from sklearn import metrics
import tensorflow as tf
import matplotlib.pyplot as plt

from supervised_models import SupervisedGraphsage
from models import SAGEInfo
from minibatch import NodeMinibatchIterator
from neigh_samplers import UniformNeighborSampler
from utils import load_data

sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))
from TimerCounter import Timer

# DISCLAIMER:
# This code file is derived from https://github.com/williamleif/GraphSAGE,
# which is under an identical MIT license as graph_confrec.


class SupervisedModel:

    def __init__(self, train_prefix, model_name, model_size="small",
                 learning_rate=0.001, epochs=10, dropout=0.0,
                 weight_decay=0.0, max_degree=100, samples_1=25, samples_2=10,
                 samples_3=0, dim_1=128, dim_2=128, batch_size=512,
                 sigmoid=False, identity_dim=0,
                 base_log_dir='../../../data/processed/graphsage/',
                 validate_iter=5000, validate_batch_size=256, gpu=0,
                 print_every=5, max_total_steps=10**10,
                 log_device_placement=False):

        self.train_prefix = train_prefix
        self.model_name = model_name
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
        self.batch_size = batch_size
        self.sigmoid = sigmoid
        self.identity_dim = identity_dim
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

    def _calc_f1(self, y_true, y_pred):
        if not self.sigmoid:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0
        return metrics.f1_score(
                y_true, y_pred, average="micro"), metrics.f1_score(
                        y_true, y_pred, average="macro")

    def _log_dir(self):
        log_dir = self.base_log_dir + \
                  self.train_prefix.rsplit("/", maxsplit=1)[-2] + "/supervised"
        log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
                    model=self.model_name, model_size=self.model_size,
                    lr=self.learning_rate)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    # Define model evaluation function
    def _evaluate(self, sess, model, minibatch_iter, size=None):
        t_test = time.time()
        feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict_val)
        mic, mac = self._calc_f1(labels, node_outs_val[0])
        return node_outs_val[1], mic, mac, (time.time() - t_test)

    def _incremental_evaluate(self, sess, model, minibatch_iter, size,
                              test=False):
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
                                     feed_dict=feed_dict_val)
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
            'labels': tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, num_classes),
                                               name='labels'),
            'batch': tf.compat.v1.placeholder(tf.int32, shape=(None),
                                              name='batch1'),
            'dropout': tf.compat.v1.placeholder_with_default(0., shape=(),
                                                   name='dropout'),
            'batch_size': tf.compat.v1.placeholder(tf.int32, name='batch_size')
            }
        return placeholders

    def _plot_losses(self, train_losses, validation_losses):
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
        plt.savefig(self._log_dir() + "losses.png", bbox_inches="tight")

    def _print_stats(self, train_losses, validation_losses, training_time):
        epochs = len(train_losses)
        time_per_epoch = training_time/epochs
        epoch_min_val = validation_losses.index(min(validation_losses))

        stats_file = self._log_dir() + "stats.txt"
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

    def _create_model(self, num_classes, placeholders, features, adj_info,
                      minibatch):
        if self.model_name == 'graphsage_mean':
            # Create model
            sampler = UniformNeighborSampler(adj_info)
            if self.samples_3 != 0:
                layer_infos = [
                        SAGEInfo("node", sampler, self.samples_1, self.dim_1),
                        SAGEInfo("node", sampler, self.samples_2, self.dim_2),
                        SAGEInfo("node", sampler, self.samples_3, self.dim_2)
                        ]
            elif self.samples_2 != 0:
                layer_infos = [
                        SAGEInfo("node", sampler, self.samples_1, self.dim_1),
                        SAGEInfo("node", sampler, self.samples_2, self.dim_2)]
            else:
                layer_infos = [
                        SAGEInfo("node", sampler, self.samples_1, self.dim_1)]

            model = SupervisedGraphsage(num_classes,
                                        placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos,
                                        weight_decay=self.weight_decay,
                                        learning_rate=self.learning_rate,
                                        model_size=self.model_size,
                                        sigmoid_loss=self.sigmoid,
                                        identity_dim=self.identity_dim,
                                        logging=True)
        elif self.model_name == 'gcn':
            # Create model
            sampler = UniformNeighborSampler(adj_info)
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
                                        aggregator_type="gcn",
                                        model_size=self.model_size,
                                        concat=False,
                                        sigmoid_loss=self.sigmoid,
                                        identity_dim=self.identity_dim,
                                        logging=True)
        elif self.model_name == 'graphsage_seq':
            sampler = UniformNeighborSampler(adj_info)
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
                                        aggregator_type="seq",
                                        model_size=self.model_size,
                                        sigmoid_loss=self.sigmoid,
                                        identity_dim=self.identity_dim,
                                        logging=True)
        elif self.model_name == 'graphsage_maxpool':
            sampler = UniformNeighborSampler(adj_info)
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
                                        aggregator_type="maxpool",
                                        model_size=self.model_size,
                                        sigmoid_loss=self.sigmoid,
                                        identity_dim=self.identity_dim,
                                        logging=True)
        elif self.model_name == 'graphsage_meanpool':
            sampler = UniformNeighborSampler(adj_info)
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
                                        aggregator_type="meanpool",
                                        model_size=self.model_size,
                                        sigmoid_loss=self.sigmoid,
                                        identity_dim=self.identity_dim,
                                        logging=True)
        else:
            raise Exception('Error: model name unrecognized.')
        return model

    def train(self, train_data, test_data=None):
        print("Training model...")
        timer = Timer()
        timer.tic()

        G = train_data[0]
        features = train_data[1]
        id_map = train_data[2]
        class_map  = train_data[4]
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
        else:
            num_classes = len(set(class_map.values()))

        if not features is None:
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

        model = self._create_model(num_classes, placeholders, features,
                                   adj_info, minibatch)

        config = tf.compat.v1.ConfigProto(
                log_device_placement=self.log_device_placement)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        # Initialize session
        sess = tf.compat.v1.Session(config=config)
        merged = tf.compat.v1.summary.merge_all()
#        summary_writer = tf.summary.FileWriter(self._log_dir(), sess.graph)

        # Initialize model saver
        saver = tf.compat.v1.train.Saver(max_to_keep=self.epochs)

        # Init variables
        sess.run(tf.compat.v1.global_variables_initializer(),
                 feed_dict={adj_info_ph: minibatch.adj})

        # Train model
        total_steps = 0
        avg_time = 0.0
        epoch_val_costs = []

        train_losses = []
        validation_losses = []

        train_adj_info = tf.compat.v1.assign(adj_info, minibatch.adj)
        val_adj_info = tf.compat.v1.assign(adj_info, minibatch.test_adj)

        for epoch in range(self.epochs):
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

                t = time.time()
                # Training step
                outs = sess.run(
                        [merged, model.opt_op, model.loss, model.preds],
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
                    sess.run(train_adj_info.op)
                    epoch_val_costs[-1] += val_cost
                    validation_loss_epoch.append(val_cost)

#                if total_steps % self.print_every == 0:
#                    summary_writer.add_summary(outs[0], total_steps)

                # Print results
                avg_time = (avg_time * total_steps + time.time() - t) / (
                        total_steps + 1)

                if total_steps % self.print_every == 0:
                    train_f1_mic, train_f1_mac = self._calc_f1(
                            labels, outs[-1])
                    print("Iter:", '%04d' % iter,
                          "train_loss=", "{:.5f}".format(train_cost),
                          "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                          "train_f1_mac=", "{:.5f}".format(train_f1_mac),
                          "val_loss=", "{:.5f}".format(val_cost),
                          "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                          "val_f1_mac=", "{:.5f}".format(val_f1_mac),
                          "time=", "{:.5f}".format(avg_time))

                iter += 1
                total_steps += 1

                if total_steps > self.max_total_steps:
                    break

            # Keep track of train and validation losses per epoch
            train_losses.append(sum(train_loss_epoch)/len(train_loss_epoch))
            validation_losses.append(
                    sum(validation_loss_epoch)/len(validation_loss_epoch))

            # Save model at each epoch
            print("Saving model at epoch {}.".format(epoch))
            saver.save(sess, os.path.join(
                       self._log_dir(), "model_epoch_" + str(epoch) + ".ckpt"))

            if total_steps > self.max_total_steps:
                break

        print("Optimization Finished!")

        training_time = timer.toc()
        self._plot_losses(train_losses, validation_losses)
        self._print_stats(train_losses, validation_losses, training_time)

        sess.run(val_adj_info.op)
        val_cost, val_f1_mic, val_f1_mac, duration = self._incremental_evaluate(
                sess, model, minibatch, self.batch_size)
        print("Full validation stats:",
              "loss=", "{:.5f}".format(val_cost),
              "f1_micro=", "{:.5f}".format(val_f1_mic),
              "f1_macro=", "{:.5f}".format(val_f1_mac),
              "time=", "{:.5f}".format(duration))
        with open(self._log_dir() + "val_stats.txt", "w") as fp:
            fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}".
                     format(val_cost, val_f1_mic, val_f1_mac, duration))

    def inference(self, test_data, model_checkpoint, gpu_mem_fraction=None):
        print("Inference.")
        timer = Timer()
        timer.tic()

        G = test_data[0]
        features = test_data[1]
        id_map = test_data[2]
        class_map  = test_data[4]
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
        else:
            num_classes = len(set(class_map.values()))

        if not features is None:
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

        model = self._create_model(num_classes, placeholders, features,
                                   adj_info, minibatch)

        config = tf.compat.v1.ConfigProto(
                log_device_placement=self.log_device_placement)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        # Initialize session
        sess = tf.compat.v1.Session(config=config)
        merged = tf.compat.v1.summary.merge_all()
#        summary_writer = tf.summary.FileWriter(self._log_dir(), sess.graph)

        # Initialize model saver
        saver = tf.compat.v1.train.Saver(max_to_keep=self.epochs)

        # Init variables
        sess.run(tf.compat.v1.global_variables_initializer(),
                 feed_dict={adj_info_ph: minibatch.adj})

        # Restore model
        print("Restoring trained model.")
        checkpoint_file = os.path.join(self._log_dir(), model_checkpoint)
        ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_file)
        if checkpoint_file:
            saver.restore(sess, checkpoint_file)
            print("Model restored.")
        else:
            print("This model checkpoint does not exist. The model might " +
                  "not be trained yet or the checkpoint is invalid.")

        val_adj_info = tf.compat.v1.assign(adj_info, minibatch.test_adj)
        sess.run(val_adj_info.op)

        print("Computing predictions...")
        t_test = time.time()
        finished = False
        val_losses = []
        val_preds = []
        nodes = []
        iter_num = 0
        while not finished:
            feed_dict_val, _, finished, nodes_subset = minibatch.incremental_node_val_feed_dict(
                    self.batch_size, iter_num, test=True)
            node_outs_val = sess.run([model.preds, model.loss],
                                     feed_dict=feed_dict_val)
            val_preds.append(node_outs_val[0])
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

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for unsupervised GraphSAGE model.')
        parser.add_argument('train_prefix',
                            help='Name of the object file that stores the '
                            + 'training data.')
        parser.add_argument("model_name",
                            choices=["graphsage_mean", "gcn", "graphsage_seq",
                                     "graphsage_maxpool", "graphsage_meanpool"
                                     ],
                            help="Model names.")
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
                            default=128,
                            help='Size of output dim ' +
                            '(final is 2x this, if using concat)')
        parser.add_argument('--dim_2',
                            type=int,
                            default=128,
                            help='Size of output dim ' +
                            '(final is 2x this, if using concat)')
        parser.add_argument('--batch_size',
                            type=int,
                            default=512,
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
        parser.add_argument('--save_embeddings',
                            action="store_false",
                            default=True,
                            help='Whether to save embeddings for all nodes ' +
                            'after training')
        parser.add_argument('--base_log_dir',
                            default='../../../data/processed/graphsage/',
                            help='Base directory for logging and saving ' +
                            'embeddings')
        parser.add_argument('--validate_iter',
                            type=int,
                            default=5000,
                            help='How often to run a validation minibatch.')
        parser.add_argument('--validate_batch_size',
                            type=int,
                            default=256,
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
        from supervised_model import SupervisedModel
        model = SupervisedModel(args.train_prefix, args.model_name,
                                args.model_size, args.learning_rate,
                                args.epochs, args.dropout, args.weight_decay,
                                args.max_degree, args.samples_1,
                                args.samples_2, args.samples_3, args.dim_1,
                                args.dim_2, args.batch_size, args.sigmoid,
                                args.identity_dim, args.base_log_dir,
                                args.validate_iter, args.validate_batch_size,
                                args.gpu, args.print_every,
                                args.max_total_steps,
                                args.log_device_placement)
        model.train(train_data)
        print("Finished.")

    if __name__ == "__main__":
        main()
