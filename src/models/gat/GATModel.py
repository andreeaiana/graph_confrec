# -*- coding: utf-8 -*-
import os
import sys
import json
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from networkx.readwrite import json_graph
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "data"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))
from TimerCounter import Timer
from AbstractClasses import AbstractModel
from preprocess_data import Processor
from model import Model as GAT
from DataLoader import DataLoader


class GATModel(AbstractModel):

    def __init__(self, embedding_type, dataset, graph_type="directed",
                 hid_units=[256, 256], n_heads=[4, 4, 1], learning_rate=0.005,
                 weight_decay=0, epochs=100000, batch_size=1, patience=100,
                 residual=False, nonlinearity=tf.nn.elu, sparse=False,
                 ffd_drop=0, attn_drop=0, gpu=0, recs=10):

        self.embedding_type = embedding_type
        self.dataset = dataset
        self.graph_type = graph_type
        self.recs = recs

        self.gat_model = GAT(self.embedding_type, self.dataset,
                             self.graph_type, hid_units, n_heads,
                             learning_rate, weight_decay, epochs, batch_size,
                             patience, residual, nonlinearity, sparse,
                             ffd_drop, attn_drop, None)
        self.preprocessor = Processor(self.embedding_type, self.dataset,
                                      self.graph_type, gpu)
        self.training_data = self._load_training_data()

        if not self._load_label_encoder():
            print("The label encoder does not exist.")

    def query_single(self, query):
        """Queries the model and returns a list of recommendations.

        Args:
            query (list): The query as needed by the model, is in the form
            [chapter_title, chapter_abstract, list(chapter_citations)].

        Returns
            list: ids of the conferences
            double: confidence scores
        """
        # Generate an ID for the query
        if len(query) < 3:
            raise ValueError("The input does not contain enough data; " +
                             "chapter title, chapter abstract, and chapter " +
                             "citations are required.")
        return self.query_batch([(query[0], query[1], query[2])])

    def query_batch(self, batch):
        """Queries the model and returns a lis of recommendations.

        Args:
            batch (list of ntuples): The list of queries as needed
            by the model. The ntuples are in the form (chapter_id,
            chapter_title, chapter_abstract, list(chapter_citations)).

        Returns
            list: ids of the conferences
            double: confidence scores
        """
        if len(batch) == 3:
            df_test = pd.DataFrame(batch,
                                   columns=["chapter_title",
                                            "chapter_abstract",
                                            "chapter_citations"])
        else:
            df_test_extended = pd.DataFrame(batch,
                                            columns=["chapter",
                                                     "chapter_title",
                                                     "chapter_abstract",
                                                     "chapter_citations"])
            df_test = df_test_extended[["chapter_title", "chapter_abstract",
                                        "chapter_citations"]]
        train_features, train_labels, train_val_features, train_val_labels, graph = self.training_data

        # Reindex test dataframe such that indices follow those from the
        # train data
        df_test.index = range(train_val_features.shape[0],
                              train_val_features.shape[0] + len(df_test))

        # Preprocess the data
        test_data = self.preprocessor.test_data(
                df_test, train_features, train_labels, train_val_features,
                train_val_labels, graph)

        # Inference on test data
        predictions = self.gat_model.test(test_data).numpy()[0][
                train_val_features.shape[0]:]

        # Compute predictions
        sorted_predictions = (-predictions).argsort(axis=1)
        conferences = list()
        confidences = list()

        for i in range(len(predictions)):
            one_hot_preds = np.zeros((self.recs, len(predictions[0])),
                                     dtype=int)
            for j in range(self.recs):
                one_hot_preds[j][sorted_predictions[i, j]] = 1
            conferences.append(list(self.label_encoder.inverse_transform(
                    one_hot_preds).flatten()))
            confidences.append(list(
                    predictions[i, sorted_predictions[:, :self.recs][i]]))

        results = [conferences, confidences]
        return results

    def train(self):
        pass

    def _load_training_data(self):
        print("Loading training data.")
        path_persistent = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "gat",
                self.embedding_type, self.dataset)
        if self.graph_type == "directed":
            names = ['x', 'y', 'allx', 'ally', 'graph_directed']
        else:
            names = ['x', 'y', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(path_persistent + "/ind.{}.{}".format(
                    self.dataset, names[i]), 'rb') as f:
                objects.append(pickle.load(f, encoding='latin1'))
        x, y, allx, ally, graph = tuple(objects)
        return x, y, allx, ally, graph
        print("Loaded.")

    def _load_label_encoder(self):
        label_encoder_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "gat",
                self.embedding_type, self.dataset, "label_encoder.pkl")
        if os.path.isfile(label_encoder_file):
            with open(label_encoder_file, "rb") as f:
                print("Loading label encoder.")
                self.label_encoder = pickle.load(f)
            print("Loaded.")
            return True
        return False
