# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
from constructor import get_placeholder, get_model, format_data, get_optimizer
from constructor import update


class ARGAModel:
    def __init__(self, model, embedding_type, dataset, hidden1=32, hidden2=16,
                 hidden3=64, discriminator_learning_rate=0.001,
                 learning_rate=0.001, weight_decay=0, dropout=0,
                 use_features=1, seed=50, epochs=200):
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        tf.compat.v1.disable_eager_execution()

        self.model = model
        self.embedding_type = embedding_type
        self.dataset = dataset
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.discriminator_learning_rate = discriminator_learning_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.use_features = use_features
        self.epochs = epochs
        self._get_folder()

        print('Model: ' + self.model)
        print("\tEmbedding: {}, Dataset: {}".format(self.embedding_type,
              self.dataset))
        print("\tHidden units layer 1: {}".format(self.hidden1))
        print("\tHidden units layer 2: {}".format(self.hidden2))
        print("\tHidden units layer 3: {}".format(self.hidden3))
        print("\tDiscriminator learning rate: {}".format(
                 self.discriminator_learning_rate))
        print("\tLearning rate: {}".format(self.learning_rate))
        print("\tWeight decay: {}".format(self.weight_decay))
        print("\tDropout: {}\n".format(self.dropout))

    def train(self):
        # Formatted data
        print("Formating the data...")
        feas = format_data(self.embedding_type, self.dataset,
                               self.use_features)
        print("Formatted.")

        # Define placeholders
        placeholders = get_placeholder(feas['adj'], self.hidden2)

        # Construct model
        print("Constructing model...")
        d_real, discriminator, ae_model = get_model(
                self.model, placeholders, feas['num_features'],
                feas['num_nodes'], feas['features_nonzero'], self.hidden1,
                self.hidden2, self.hidden3)
        print("Constructed.")

        # Optimizer
        print("Initializing optimizer...")
        opt = get_optimizer(
                self.model, ae_model, discriminator, placeholders,
                feas['pos_weight'], feas['norm'], d_real, feas['num_nodes'],
                self.discriminator_learning_rate, self.learning_rate)
        print("Initialized.")

        # Initialize session
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        # Train model
        print("Training model...")
        for epoch in range(self.epochs):
            print("Epoch {}".format(epoch+1))
            embeddings, avg_cost = update(
                    ae_model, opt, sess, feas['adj_norm'], feas['adj_label'],
                    feas['features'], placeholders, feas['adj'], self.dropout,
                    self.hidden2)
            print("AVG cost at epoch {}: {}".format(epoch, avg_cost))
        print("Trained.")
        self._save_embeddings(embeddings)

    def inference(self):
        pass

    def _embeddings_file(self):
        file = "arga_embeddings_" + self.model + "_" + str(self.hidden1) \
                + "_" + str(self.hidden2) + "_" + str(self.hidden3) + "_" + \
                str(self.discriminator_learning_rate) + "_" + \
                str(self.learning_rate) + "_" + str(self.weight_decay) +\
                "_" + str(self.dropout) + ".pkl"
        path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "scibert_arga",
                self.embedding_type, self.dataset, file)
        return path

    def _save_embeddings(self, embeddings):
        print("Saving embeddings to disk...")
        file_embeddings = self._embeddings_file()
        with open(file_embeddings, "wb") as f:
            pickle.dump(embeddings, f)
        print("Saved.")

    def _load_embeddings(self):
        file_embeddings = self._embeddings_file()
        with open(embeddings_file, "rb") as f:
            embeddings = pickle.load(f)
        return embeddings

    def _get_folder(self):
        model_dir = self.model + "_" + str(self.hidden1) + "_" + \
                    str(self.hidden2) + "_" + str(self.hidden3) + "_" + \
                    str(self.discriminator_learning_rate) + "_" + \
                    str(self.learning_rate) + "_" + str(self.weight_decay) +\
                    "_" + str(self.dropout)
        self.path_persistent = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "processed", "scibert_arga",
                "arga_models", self.embedding_type, self.dataset, model_dir)
        if not os.path.exists(self.path_persistent):
            os.makedirs(self.path_persistent)

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for ARGA model.')
        parser.add_argument('model',
                            choices=["arga_ae", "arga_vae"],
                            help="Type of model.")
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
        parser.add_argument("--hidden1",
                            type=int,
                            default=32,
                            help="Number of units in hidden layer 1.")
        parser.add_argument("--hidden2",
                            type=int,
                            default=16,
                            help="Number of units in hidden layer 2.")
        parser.add_argument("--hidden3",
                            type=int,
                            default=64,
                            help="Number of units in hidden layer 3.")
        parser.add_argument("--discriminator_learning_rate",
                            type=float,
                            default=0.001,
                            help="Initial learning rate of the discriminator.")
        parser.add_argument("--learning_rate",
                            type=float,
                            default=0.001,
                            help="Initial learning rate.")
        parser.add_argument("--weight_decay",
                            type=float,
                            default=0,
                            help="Weight for L2 loss on embedding matrix.")
        parser.add_argument("--dropout",
                            type=float,
                            default=0,
                            help="Dropout rate (1 - keep probability).")
        parser.add_argument("--use_features",
                            type=int,
                            default=1,
                            help="Whether to use features (1) or not (0).")
        parser.add_argument("--seed",
                            type=int,
                            default=50,
                            help="Seed for fixing the results.")
        parser.add_argument("--epochs",
                            type=int,
                            default=200,
                            help="Number of epochs.")
        args = parser.parse_args()

        print("Starting...")
        from arga_model import ARGAModel
        model = ARGAModel(args.model, args.embedding_type, args.dataset,
                          args.hidden1, args.hidden2, args.hidden3,
                          args.discriminator_learning_rate, args.learning_rate,
                          args.weight_decay, args.dropout, args.use_features,
                          args.seed, args.epochs)
        model.train()
        print("Finished.")

    if __name__ == "__main__":
        main()









