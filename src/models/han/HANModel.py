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
from model import Model as HAN
from DataLoader import DataLoader


class HANModel(AbstractModel):

    def __init__(self, model, embedding_type, hid_units=[64], n_heads=[8, 1],
                 learning_rate=0.005, weight_decay=0, epochs=10000,
                 batch_size=1, patience=100, residual=False,
                 nonlinearity=tf.nn.elu, ffd_drop=0.5, attn_drop=0.5, gpu=0,
                 recs=10):

        self.embedding_type = embedding_type
        self.recs = recs

        self.han_model = HAN(model, self.embedding_type, hid_units, n_heads,
                             learning_rate, weight_decay, epochs, batch_size,
                             patience, residual, nonlinearity, ffd_drop,
                             attn_drop, None)
        self.preprocessor = Processor(self.embedding_type, gpu)
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
        if len(query) < 4:
            raise ValueError("The input does not contain enough data; " +
                             "chapter title, chapter abstract, chapter " +
                             "citations, and chapter authors are required."
                             )
        query_id = "new_node_id:" + "-".join(
                [str(i) for i in random.sample(range(0, 10000), 5)])
        authors_df = pd.DataFrame({"author_name": query[3],
                                  "chapter": [query_id]*len(query[3])})
        return self.query_batch([(query_id, query[0], query[1], query[2])],
                                authors_df)

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
        if len(batch) == 4:
            df_test = pd.DataFrame(batch[0],
                                   columns=["chapter_title",
                                            "chapter_abstract",
                                            "chapter_citations"])
        else:
            df_test_extended = pd.DataFrame(batch[0],
                                            columns=["chapter",
                                                     "chapter_title",
                                                     "chapter_abstract",
                                                     "chapter_citations"])
            df_test = df_test_extended[["chapter_title",
                                        "chapter_abstract",
                                        "chapter_citations"]]
        authors_df = batch[1]

        train_idx, val_idx, features, labels, PAP, PCP = self.training_data

        # Reindex test dataframe such that indices follow those from the
        # train data
        df_test.index = range(len(train_idx[0]) + len(val_idx[0]),
                              len(train_idx[0]) + len(val_idx[0]) +
                              len(df_test))

        # Preprocess the data
        test_data = self.preprocessor.test_data(
                df_test, authors_df, train_idx, features, labels, PCP, PAP)

        # Inference on test data
        predictions = self.han_model.test(test_data).numpy()
        predictions = predictions[0][len(train_idx[0])+len(val_idx[0]):]

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
                "..", "..", "..", "data", "interim", "han",
                self.embedding_type)

        names = ["train_idx", "val_idx", "features", "labels", "PAP", "PCP"]
        objects = []
        for i in range(len(names)):
            with open(os.path.join(path_persistent, "{}.pkl".format(names[i])),
                      "rb") as f:
                if sys.version_info > (3, 0):
                    objects.append(pickle.load(f, encoding="latin1"))
                else:
                    objects.append(pickle.load(f))
        train_idx, val_idx, features, labels, PAP, PCP = tuple(objects)
        print("Loaded.")
        return train_idx, val_idx, features, labels, PAP, PCP

    def _load_label_encoder(self):
        label_encoder_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "han",
                self.embedding_type, "label_encoder.pkl")
        if os.path.isfile(label_encoder_file):
            with open(label_encoder_file, "rb") as f:
                print("Loading label encoder.")
                self.label_encoder = pickle.load(f)
            print("Loaded.")
            return True
        return False
