# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import tensorflow as tf

from gat import GAT
from process import load_data, preprocess_features
from process import adj_to_bias, preprocess_adj_bias

# DISCLAIMER:
# This code file is derived from https://github.com/PetarV-/GAT,
# which is under an identical MIT license as graph_confrec.
# Conversion of original code to TF 2.0 is inspired by
# https://github.com/calciver/Graph-Attention-Networks/blob/master/Tensorflow_2_0_Graph_Attention_Networks_(GAT).ipynb


class GATModel:

    def __init__(self, embedding_type, dataset, hid_units=[8], n_heads=[8,1],
                 learning_rate=0.005, weight_decay=0.0005, epochs=100000,
                 batch_size=1, patience=100, residual=False,
                 nonlinearity=tf.nn.elu, sparse=False, ffd_drop=0.6,
                 attn_drop=0.6, gpu=0):

        print("Initiating, using gpu {}.".format(gpu))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        self.embedding_type = embedding_type
        self.dataset = dataset
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.residual = residual
        if nonlinearity is not None:
            self.nonlinearity = nonlinearity
        else:
            self.nonlinearity = tf.nn.elu
        self.Sparse = sparse
        self.ffd_drop = ffd_drop
        self.attn_drop = attn_drop
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.path_persistent = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "processed", "gat",
                self.embedding_type, self.dataset)
        if not os.path.isdir(self.path_persistent):
            os.mkdir(self.path_persistent)

        print('Model: ' + str('SpGAT' if self.Sparse else 'GAT'))
        print("Dataset: {}, Embedding: {}".format(self.dataset,
              self.embedding_type))
        print("----- Opt. hyperparameters -----")
        print("\tLearning rate: {}".format(self.learning_rate))
        print("\tWeight decay: {}".format(self.weight_decay))
        print("----- Archi. hyperparameters -----")
        print("\tNumber of layers: {}".format(len(self.hid_units)))
        print("\tNumber of units per layer: {}".format(self.hid_units))
        print("\tNumber of attention heads: {}".format(self.n_heads))
        print("\tResidual: {}".format(self.residual))
        print("\tNonlinearity: {}\n".format(self.nonlinearity))

    def _train(self, model, inputs, bias_mat, lbl_in, msk_in):
        with tf.GradientTape() as tape:
            logits, accuracy, loss = model(inputs=inputs,
                                           bias_mat=bias_mat,
                                           lbl_in=lbl_in,
                                           msk_in=msk_in,
                                           training=True)
        gradients = tape.gradient(loss, model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
        self.optimizer.apply_gradient(gradient_variables)

        return logits, accuracy, loss

    def evaluate(model, inputs, bias_mat, lbl_in, msk_in):
        logits, accuracy, loss = model(inputs=inputs,
                                       bias_mat=bias_mat,
                                       lbl_in=lbl_in,
                                       msk_in=msk_in,
                                       training=False)
        return logits, accuracy, loss

    def train(self):
        print("Loading data...")
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
                self.embedding_type, self.dataset)
        print("Loaded.\n")

        features, spars = preprocess_features(features)

        nb_nodes = features.shape[0]
        ft_size = features.shape[1]
        nb_classes = y_train.shape[1]

        features = features[np.newaxis]
        y_train = y_train[np.newaxis]
        y_val = y_val[np.newaxis]
        y_test = y_test[np.newaxis]
        train_mask = train_mask[np.newaxis]
        val_mask = val_mask[np.newaxis]
        test_mask = test_mask[np.newaxis]

        print("Parameters: batch size={}, nb_nodes={}, ft_size={}, nb_classes={}".format(
                self.batch_size, nb_nodes, ft_size, nb_classes))

        if self.Sparse:
            biases = preprocess_adj_bias(adj)
        else:
            adj = adj.todense()
            adj = adj[np.newaxis]
            biases = adj_to_bias(adj, [nb_nodes], nhood=1)

        model = GAT(self.hid_units, self.n_heads, nb_classes, nb_nodes,
                    self.Sparse, ffd_drop=self.ffd_drop,
                    attn_drop=self.attn_drop, activation=self.nonlinearity,
                    residual=self.residual)

        vlss_mn = np.inf
        vacc_mx = 0.0
        curr_step = 0

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        # Training the model
        for epoch in range(self.epochs):
            print("###############  Epoch {}  ###############".format(epoch))
            # Training
            tr_step = 0
            tr_size = features.shape[0]
            while tr_step * self.batch_size < tr_size:
                if self.Sparse:
                    bbias = biases
                else:
                    bbias = biases[tr_step + self.batch_size: (
                            tr_step+1)*self.batch_size]
                _, acc_tr, loss_value_tr = self._train(
                        model,
                        inputs=features[tr_step*self.batch_size: (
                                tr_step+1)*self.batch_size],
                        bias_mat=bbias,
                        lbl_in=y_train[tr_step*self.batch_size: (
                                tr_step+1)*self.batch_size],
                        msk_in=train_mask[tr_step*self.batch_size: (
                                tr_step+1)*self.batch_size])
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            # Validation
            vl_step = 0
            vl_size = features.shape[0]
            while vl_step * self.batch_size < vl_size:
                if self.Sparse:
                    bbias = biases
                else:
                    bbias = biases[vl_step*self.batch_size: (
                            vl_step+1)*self.batch_size]
                _, acc_vl, loss_value_vl = self.evaluate(
                        model,
                        inputs=features[vl_step*self.batch_size: (
                                vl_step+1)*self.batch_size],
                        bias_mat=bbias,
                        lbl_in=y_val[vl_step*self.batch_size: (
                                vl_step+1)*self.batch_size],
                        msk_in=val_mask[vl_step*self.batch_size: (
                                vl_step+1)*self.batch_size])
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            print("Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f" % (
                    train_loss_avg/tr_step, train_acc_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step))

            # Early stopping
            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    working_weights = model.get_weights()
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == self.patience:
                    print("Early stop! Min loss: {}, Max accuracy: {}".format(
                            vlss_mn, vacc_mx))
                    print("Early stop model validation loass: {}, accuracy: {}".format(
                            vlss_early_model, vacc_early_model))
                    model.set_weights(working_weights)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

            # Save model
            model.save_weights(self.path_persistent + '/model',
                               save_format='tf')

    def test(self):
        print("Loading data...")
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(self.dataset)
        print("Loaded.\n")

        features, spars = preprocess_features(features)

        nb_nodes = features.shape[0]
        ft_size = features.shape[1]
        nb_classes = y_train.shape[1]

        features = features[np.newaxis]
        y_test = y_test[np.newaxis]
        test_mask = test_mask[np.newaxis]

        print("Parameters: batch size={}, nb_nodes={}, ft_size={}, nb_classes={}".format(
                self.batch_size, nb_nodes, ft_size, nb_classes))

        if self.Sparse:
            biases = preprocess_adj_bias(adj)
        else:
            adj = adj.todense()
            adj = adj[np.newaxis]
            biases = adj_to_bias(adj, [nb_nodes], nhood=1)

        model = GAT(self.hid_units, self.n_heads, nb_classes, nb_nodes,
                    self.Sparse, ffd_drop=self.ffd_drop,
                    attn_drop=self.attn_drop, activation=self.nonlinearity,
                    residual=self.residual)

        # Restore model weights
        try:
            print("Loading model weights...")
            model.load_weights(self.path_persistent + "/model")
            print("Model weights restored.")
        except Exception as e:
            print("Failed loading model weights: {}".format(e))

        ts_step = 0
        ts_size = features.shape[0]
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * self.batch_size < ts_size:
            if self.Sparse:
                bbias = biases
            else:
                bbias = biases[ts_step*self.batch_size:(
                        ts_step + 1)*self.batch_size]

            logits, acc_ts, loss_value_ts = self.evaluate(
                    model,
                    inputs=features[ts_step * self.batch_size:(
                            ts_step + 1)*self.batch_size],
                    bias_mat=bbias,
                    lbl_in=y_test[ts_step * self.batch_size:(
                            ts_step + 1)*self.batch_size],
                    msk_in=test_mask[ts_step*self.batch_size:(
                            ts_step + 1)*self.batch_size])
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        predictions = tf.nn.softmax(logits)
        return predictions

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for unsupervised GraphSAGE model.')
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
        parser.add_argument("--hid_units",
                            type=int,
                            nargs="+",
                            default=[8],
                            help="Number of hidden units per each attention "
                            + "head in each layer.")
        parser.add_argument('--n_heads',
                            type=int,
                            nargs="+",
                            default=[8, 1],
                            help='Additional entry for the output layer.')
        parser.add_argument('--learning_rate',
                            type=float,
                            default=0.005,
                            help='Learning rate.')
        parser.add_argument('--weight_decay',
                            type=float,
                            default=0.0005,
                            help='Weight decay.')
        parser.add_argument('--epochs',
                            type=int,
                            default=100000,
                            help='Number of epochs to train.')
        parser.add_argument('--batch_size',
                            type=int,
                            default=1,
                            help='Batch size.')
        parser.add_argument('--patience',
                            type=int,
                            default=100)
        parser.add_argument('--residual',
                            action="store_true",
                            default=False)
        parser.add_argument('--nonlinearity',
                            help="Type of activation used")
        parser.add_argument('--sparse',
                            action='store_true',
                            default=False,
                            help="Whether to use the sparse model version")
        parser.add_argument('--ffd_drop',
                            type=float,
                            default=0.6)
        parser.add_argument('--attn_drop',
                            type=float,
                            default=0.6)
        parser.add_argument('--gpu',
                            type=int,
                            default=0,
                            help='Which gpu to use.')
        args = parser.parse_args()

        print("Starting...")
        from model_train import GATModel
        model = GATModel(args.embedding_type, args.dataset, args.hid_units,
                         args.n_heads, args.learning_rate, args.weight_decay,
                         args.epochs, args.batch_size, args.patience,
                         args.residual, args.nonlinearity, args.sparse,
                         args.ffd_drop, args.attn_drop, args.gpu)
        model.train()
        print("Finished.")

    if __name__ == "__main__":
        main()
