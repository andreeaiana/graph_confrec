# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

from inits import *
from sampler import *
from models import GCNAdapt, GCNAdaptMix

# DISCLAIMER:
# This code file is derived from https://github.com/huangwb/AS-GCN


class Model:
    def __init__(self, embedding_type, dataset, model_name, max_degree=696,
                 learning_rate=0.001, weight_decay=5e-4, dropout=0.0,
                 epochs=300, early_stopping=30, hidden1=16, rank=128, skip=0,
                 var=0.5, sampler_device="cpu", gpu=None):

        print("Initiating, using gpu {}.\n".format(gpu))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.embedding_type = embedding_type
        self.dataset = dataset
        self.model_name = model_name
        self.max_degree = max_degree
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.hidden1 = hidden1
        self.rank = rank
        self.skip = skip
        self.var = var
        self.sampler_device = sampler_device

        # Set random seed
        seed = 123
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        tf.compat.v1.disable_eager_execution()

    # Define model evaluation function
    def _evaluate(self, sess, model, features, support, prob_norm, labels,
                  mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict_with_prob(
                features, support, prob_norm, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy],
                            feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    def train(self):
        print("Preprocessing data...\n")
        adj, adj_train, adj_val_train, features, train_features, y_train, y_val, val_index = prepare_data(
                self.embedding_type, self.dataset, self.max_degree)
        print("Preprocessed.\n")

        rank1 = self.rank
        rank0 = self.rank
        num_train = adj_train.shape[0] - 1
        input_dim = features.shape[1]
        scope = "test"

        if self.model_name == "gcn_adapt_mix":
            num_supports = 1
            propagator = GCNAdaptMix
            val_supports = [sparse_to_tuple(adj[val_index, :])]
            val_features = [features, features[val_index, :]]
            val_probs = [np.ones(adj.shape[0])]
            layer_sizes = [rank1, 256]
        elif self.model_name == "gcn_adapt":
            num_supports = 2
            propagator = GCNAdapt
            val_supports = [sparse_to_tuple(adj),
                            sparse_to_tuple(adj[val_index, :])]
            val_features = [features, features, features[val_index, :]]
            val_probs = [np.ones(adj.shape[0]), np.ones(adj.shape[0])]
            layer_sizes = [rank0, rank1, 256]
        else:
            raise ValueError('Invalid argument for model: ' + str(
                    self.model_name))

        # Define placeholders
        placeholders = {
            'batch': tf.compat.v1.placeholder(tf.int32),
            'adj': tf.compat.v1.placeholder(
                    tf.int32, shape=(num_train+1, self.max_degree)),
            'adj_val': tf.compat.v1.placeholder(
                    tf.float32, shape=(num_train+1, self.max_degree)),
            'features': tf.compat.v1.placeholder(
                    tf.float32, shape=train_features.shape),
            'support': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in
                        range(num_supports)],
            'prob': [tf.compat.v1.placeholder(tf.float32) for _ in range(
                    num_supports)],
            'features_inputs': [tf.compat.v1.placeholder(
                    tf.float32, shape=(None, input_dim)) for _ in range(
                            num_supports+1)],
            'labels': tf.compat.v1.placeholder(
                    tf.float32, shape=(None, y_train.shape[1])),
            'labels_mask': tf.compat.v1.placeholder(tf.int32),
            'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
            # helper variable for sparse dropout
            'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)
        }

        # Sampling parameters shared by the sampler and model
        with tf.compat.v1.variable_scope(scope):
            w_s = glorot([features.shape[-1], 2], name='sample_weights')

        # Create sampler
        if self.sampler_device == 'cpu':
            with tf.device('/cpu:0'):
                sampler_tf = SamplerAdapt(
                        placeholders, input_dim=input_dim,
                        layer_sizes=layer_sizes, scope=scope)
                features_sampled, support_sampled, p_u_sampled = sampler_tf.sampling(
                        placeholders['batch'])
        else:
            sampler_tf = SamplerAdapt(
                    placeholders, input_dim=input_dim, layer_sizes=layer_sizes,
                    scope=scope)
            features_sampled, support_sampled, p_u_sampled = sampler_tf.sampling(
                    placeholders['batch'])

        # Create model
        model = propagator(placeholders,
                           input_dim=input_dim,
                           learning_rate=self.learning_rate,
                           weight_decay=self.weight_decay,
                           hidden1=self.hidden1,
                           var=self.var,
                           logging=True,
                           name=scope)

        # Initialize session
        config = tf.compat.v1.ConfigProto(device_count={"CPU": 1},
                                          inter_op_parallelism_threads=0,
                                          intra_op_parallelism_threads=0,
                                          allow_soft_placement=True,
                                          log_device_placement=False)
        sess = tf.compat.v1.Session(config=config)

        # Init variables
        sess.run(tf.compat.v1.global_variables_initializer(),
                 feed_dict={placeholders['adj']: adj_train,
                            placeholders['adj_val']: adj_val_train,
                            placeholders['features']: train_features})

        # Prepare training
        saver = tf.compat.v1.train.Saver()
        save_dir = self._save_dir()
        acc_val = []
        acc_train = []
        losses_val = []
        losses_train = []
        train_time = []
        train_time_sample = []
        max_acc = 0
        t = time.time()

        # Train model
        print("Training model...")
        for epoch in range(self.epochs):
            sample_time = 0
            t1 = time.time()
            for batch in iterate_minibatches_listinputs(
                    [y_train, np.arange(num_train)], batchsize=256,
                    shuffle=True):
                [y_train_batch, train_batch] = batch

                if sum(train_batch) < 1:
                    continue
                ts = time.time()
                features_inputs, supports, probs = sess.run(
                        [features_sampled, support_sampled, p_u_sampled],
                        feed_dict={placeholders['batch']: train_batch})
                sample_time += time.time()-ts

                # Construct feed dictionary
                feed_dict = construct_feed_dict_with_prob(
                        features_inputs, supports, probs, y_train_batch, [],
                        placeholders)
                feed_dict.update({placeholders['dropout']: self.dropout})

                # Training step
                outs = sess.run([model.opt_op, model.loss, model.accuracy],
                                feed_dict=feed_dict)
                acc_train.append(outs[-2])
                losses_train.append(outs[1])

            train_time_sample.append(time.time() - t1)
            train_time.append(time.time() - t1 - sample_time)

            # Validation
            cost, acc, duration = self._evaluate(
                    sess, model, val_features, val_supports, val_probs, y_val,
                    [], placeholders)
            acc_val.append(acc)
            losses_val.append(cost)

            # Print results
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]),
                  "val_loss=", "{:.5f}".format(cost),
                  "val_acc=", "{:.5f}".format(acc),
                  "time=", "{:.5f}".format(train_time_sample[epoch]),
                  "\n")

            if acc > max_acc:
                max_acc = acc
                print("Saving model at epoch: {}, accuracy: {}, loss: {}.".format(
                        epoch+1, acc, cost))
                saver.save(sess, os.path.join(save_dir, "model.ckpt"))
                print("Saved.\n")

        print("Training finished.\n")
        self._plot_losses(losses_train, losses_val)
        self._plot_accuracies(acc_train, acc_val)
        self._print_stats(losses_train, losses_val, acc_train, acc_val,
                          train_time_sample)

    def test(self, test_data):
        # Prepare data
        adj, adj_train, adj_val_train, features, train_features, y_train, y_test, test_index = test_data

        rank1 = self.rank
        rank0 = self.rank
        num_train = adj_train.shape[0] - 1
        input_dim = features.shape[1]
        scope = 'test'

        if self.model_name == 'gcn_adapt_mix':
            num_supports = 1
            propagator = GCNAdaptMix
            test_supports = [sparse_to_tuple(adj[test_index, :])]
            test_features = [features, features[test_index, :]]
            test_probs = [np.ones(adj.shape[0])]
            layer_sizes = [rank1, 256]
        elif self.model_name == 'gcn_adapt':
            num_supports = 2
            propagator = GCNAdapt
            test_supports = [sparse_to_tuple(adj),
                             sparse_to_tuple(adj[test_index, :])]
            test_features = [features, features, features[test_index, :]]
            test_probs = [np.ones(adj.shape[0]), np.ones(adj.shape[0])]
            layer_sizes = [rank0, rank1, 256]
        else:
            raise ValueError('Invalid argument for model: ' + str(
                    self.model_name))

        # Define placeholders
        placeholders = {
            'batch': tf.compat.v1.placeholder(tf.int32),
            'adj': tf.compat.v1.placeholder(
                    tf.int32, shape=(num_train+1, self.max_degree)),
            'adj_val': tf.compat.v1.placeholder(
                    tf.float32, shape=(num_train+1, self.max_degree)),
            'features': tf.compat.v1.placeholder(
                    tf.float32, shape=train_features.shape),
            'support': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in
                        range(num_supports)],
            'prob': [tf.compat.v1.placeholder(tf.float32) for _ in range(
                    num_supports)],
            'features_inputs': [tf.compat.v1.placeholder(
                    tf.float32, shape=(None, input_dim)) for _ in range(
                            num_supports+1)],
            'labels': tf.compat.v1.placeholder(
                    tf.float32, shape=(None, y_train.shape[1])),
            'labels_mask': tf.compat.v1.placeholder(tf.int32),
            'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
            # helper variable for sparse dropout
            'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)
        }

        # Sampling parameters shared by the sampler and model
        with tf.compat.v1.variable_scope(scope):
            w_s = glorot([features.shape[-1], 2], name='sample_weights')

        # Create sampler
        if self.sampler_device == 'cpu':
            with tf.device('/cpu:0'):
                sampler_tf = SamplerAdapt(
                        placeholders, input_dim=input_dim,
                        layer_sizes=layer_sizes, scope=scope)
                features_sampled, support_sampled, p_u_sampled = sampler_tf.sampling(
                        placeholders['batch'])
        else:
            sampler_tf = SamplerAdapt(
                    placeholders, input_dim=input_dim, layer_sizes=layer_sizes,
                    scope=scope)
            features_sampled, support_sampled, p_u_sampled = sampler_tf.sampling(
                    placeholders['batch'])

        # Create model
        model = propagator(placeholders,
                           input_dim=input_dim,
                           learning_rate=self.learning_rate,
                           weight_decay=self.weight_decay,
                           hidden1=self.hidden1,
                           var=self.var,
                           logging=True,
                           name=scope)

        # Initialize session
        config = tf.compat.v1.ConfigProto(device_count={"CPU": 1},
                                          inter_op_parallelism_threads=0,
                                          intra_op_parallelism_threads=0,
                                          allow_soft_placement=True,
                                          log_device_placement=False)
        sess = tf.compat.v1.Session(config=config)

        # Init variables
        sess.run(tf.compat.v1.global_variables_initializer(),
                 feed_dict={placeholders['adj']: adj_train,
                            placeholders['adj_val']: adj_val_train,
                            placeholders['features']: train_features})

        # Prepare training
        saver = tf.compat.v1.train.Saver()
        save_dir = self._save_dir()

        # Testing
        print("Restoring model...")
        if os.path.exists(os.path.join(save_dir, "model.ckpt.index")):
            saver.restore(sess, os.path.join(save_dir, "model.ckpt"))
            print('Model restored.\n')
        else:
            raise ValueError("Model checkpoint does not exist.\n")

        print("Computing predictions...")
        feed_dict_val = construct_feed_dict_with_prob(
                test_features, test_supports, test_probs, y_test, [],
                placeholders)
        predictions = sess.run([model.preds], feed_dict=feed_dict_val)
        print("Computed.\n")
        return predictions[0]

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
        plt.savefig(os.path.join(self._save_dir(), "losses.png"),
                    bbox_inches="tight")

    def _plot_accuracies(self, train_accuracies, val_accuracies):
        # Plot the training and validation accuracies
        ymax = max(max(train_accuracies), max(val_accuracies))
        ymin = min(min(train_accuracies), min(val_accuracies))
        plt.plot(train_accuracies, color='tab:blue')
        plt.plot(val_accuracies, color='tab:orange')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend(["train", "validation"], loc=3)
        plt.ylim(ymin=ymin-0.5, ymax=ymax+0.5)
        plt.savefig(os.path.join(self._save_dir(), "accuracies.png"),
                    bbox_inches="tight")

    def _print_stats(self, train_losses, validation_losses, train_accuracies,
                     validation_accuracies, train_time_sample):
        epochs = len(validation_losses)
        time_per_epoch = np.mean(np.array(train_time_sample))
        total_time = np.sum(np.array(train_time_sample))
        epoch_min_val = validation_losses.index(min(validation_losses))
        epoch_max_acc = validation_accuracies.index(max(validation_accuracies))

        stats_file = os.path.join(self._save_dir(), "stats.txt")
        with open(stats_file, "w") as f:
            self._print("Total number of epochs trained: {}, average time per epoch: {} minutes.\n".format(
                    epochs, round(time_per_epoch/60, 4)), f)
            self._print("Total time trained: {} minutes.\n".format(
                    round(total_time/60, 4)), f)
            self._print("Lowest validation loss at epoch {} = {}.\n".format(
                    epoch_min_val, validation_losses[epoch_min_val]), f)
            self._print("Highest validation accuracy at epoch {} = {}.\n".format(
                    epoch_max_acc, validation_accuracies[epoch_max_acc]), f)
            f.write("\n\n")
            for epoch in range(epochs):
                f.write('Epoch: %.f | Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f\n' %
                        (epoch, train_losses[epoch], train_accuracies[epoch],
                         validation_losses[epoch], validation_accuracies[epoch]
                         ))

    def _print(self, text, f):
        print(text)
        f.write(text)

    def _save_dir(self):
        model_dir = self.model_name + "_" + str(self.learning_rate) + "_" + \
                    str(self.weight_decay) + "_" + str(self.hidden1) + "_" + \
                    str(self.rank) + "_" + str(self.skip) + "_" + str(self.var)
        path_persistent = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "processed", "as_gcn",
                self.embedding_type, self.dataset, model_dir)
        if not os.path.exists(path_persistent):
            os.makedirs(path_persistent)
        return path_persistent

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for ASGCN model.')
        parser.add_argument('embedding_type',
                            choices=["AVG_L", "AVG_2L", "AVG_SUM_L4",
                                     "AVG_SUM_ALL", "MAX_2L",
                                     "CONC_AVG_MAX_2L", "CONC_AVG_MAX_SUM_L4",
                                     "SUM_L", "SUM_2L"
                                     ],
                            help="Type of embedding.")
        parser.add_argument('dataset',
                            help='Name of the object file that stores the '
                            + 'training data.')
        parser.add_argument("model_name",
                            choices=["gcn_adapt", "gcn_adapt_mix"],
                            help="Model names.")
        parser.add_argument('--max_degree',
                            type=int,
                            default=696,
                            help='Maximum degree for constructing the ' +
                                 'adjacent matrix.')
        parser.add_argument('--learning_rate',
                            type=float,
                            default=0.001,
                            help='Learning rate.')
        parser.add_argument('--weight_decay',
                            type=float,
                            default=5e-4,
                            help='Weight decay.')
        parser.add_argument('--dropout',
                            type=float,
                            default=0.0,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--epochs',
                            type=int,
                            default=300,
                            help='Number of epochs to train.')
        parser.add_argument('--early_stopping',
                            type=int,
                            default=30,
                            help='Tolerance for early stopping (# of epochs).')
        parser.add_argument("--hidden1",
                            type=int,
                            default=16,
                            help="Number of units in hidden layer 1.")
        parser.add_argument("--rank",
                            type=int,
                            default=128,
                            help="The number of nodes per layer.")
        parser.add_argument('--skip',
                            type=float,
                            default=0,
                            help='If use skip connection.')
        parser.add_argument('--var',
                            type=float,
                            default=0.5,
                            help='If use variance reduction.')
        parser.add_argument("--sampler_device",
                            choices=["gpu", "cpu"],
                            default="cpu",
                            help="The device for sampling: cpu or gpu.")
        parser.add_argument('--gpu',
                            type=int,
                            help='Which gpu to use.')
        args = parser.parse_args()

        print("Starting...")
        from model import Model
        model = Model(args.embedding_type, args.dataset, args.model_name,
                      args.max_degree, args.learning_rate, args.weight_decay,
                      args.dropout, args.epochs, args.early_stopping,
                      args.hidden1, args.rank, args.skip, args.var,
                      args.sampler_device, args.gpu)
        model.train()
        print("Finished.")

    if __name__ == "__main__":
        main()
