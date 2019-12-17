# -*- coding: utf-8 -*-
import os
import sys
import json
import pickle
import random
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "data"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))
from TimerCounter import Timer
from AbstractClasses import AbstractModel
from preprocess_data import Processor
from model import Model as ASGCN
from DataLoader import DataLoader


class ASGCNModel(AbstractModel):

    def __init__(self, embedding_type, dataset, model_name, max_degree=696,
                 learning_rate=0.001, weight_decay=5e-4, dropout=0.0,
                 epochs=300, early_stopping=30, hidden1=16, rank=128, skip=0,
                 var=0.5, sampler_device="cpu", gpu=None, recs=10):

        self.embedding_type = embedding_type
        self.dataset = dataset
        self.model_name = model_name
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
        self.recs = recs

        self.preprocessor = Processor(self.embedding_type, self.dataset, gpu)
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
        test_data, max_degree = self.preprocessor.test_data(
                df_test, train_features, train_labels, train_val_features,
                train_val_labels, graph)

        # Inference on test data
        asgcn_model = ASGCN(
                self.embedding_type, self.dataset, self.model_name, max_degree,
                self.learning_rate, self.weight_decay, self.dropout,
                self.epochs, self.early_stopping, self.hidden1, self.rank,
                self.skip, self.var, self.sampler_device, gpu=None)
        predictions = asgcn_model.test(test_data)

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
        names = ['x', 'y', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(path_persistent + "/ind.{}.{}".format(
                    self.dataset, names[i]), 'rb') as f:
                objects.append(pickle.load(f, encoding='latin1'))
        x, y, allx, ally, graph = tuple(objects)
        print("Loaded.")
        return x, y, allx, ally, graph

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
