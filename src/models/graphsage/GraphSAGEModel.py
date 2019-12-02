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
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))
from TimerCounter import Timer
from AbstractClasses import AbstractModel
from preprocess_data import Processor
from supervised_model import SupervisedModel


class GraphSAGEModel(AbstractModel):

    def __init__(self, embedding_type, graph_type, model_checkpoint,
                 train_prefix, model_name, model_size="small",
                 learning_rate=0.001, epochs=10, dropout=0.0,
                 weight_decay=0.0, max_degree=100, samples_1=25, samples_2=10,
                 samples_3=0, dim_1=128, dim_2=128, random_context=True,
                 batch_size=512, sigmoid=False, identity_dim=0,
                 base_log_dir='../../../data/processed/graphsage/',
                 validate_iter=5000, validate_batch_size=256, gpu=0,
                 print_every=5, max_total_steps=10**10,
                 log_device_placement=False, recs=10):

        self.embedding_type = embedding_type
        self.graph_type = graph_type
        self.model_checkpoint = model_checkpoint
        self.recs = recs

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        self.graphsage_model = SupervisedModel(
                train_prefix, model_name, model_size, learning_rate, epochs,
                dropout, weight_decay, max_degree, samples_1, samples_2,
                samples_3, dim_1, dim_2, random_context, batch_size, sigmoid,
                identity_dim, base_log_dir, validate_iter, validate_batch_size,
                gpu, print_every, max_total_steps, log_device_placement)
        self.preprocessor = Processor(self.embedding_type, self.graph_type,
                                      gpu)

        if not self._load_training_graph():
            print("The training graph does not exist.")

        if not self._load_training_class_map():
            print("The training class map dows not exist.")

        if not self._load_training_walks():
            print("The walks do not exist.")

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
        query_id = "new_node_id:" + "-".join(
                [str(i) for i in random.sample(range(0, 10000), 5)])
        if len(query) < 3:
            raise ValueError("The input does not contain enough data; " +
                             "chapter title chapter abstract, and chapter " +
                             "citations are required.")
        return self.query_batch([(query_id, query[0], query[1], query[2])])

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
        df_test = pd.DataFrame(batch, columns=["chapter", "chapter_title",
                                               "chapter_abstract",
                                               "chapter_citations"])

        # Preprocess the data
        graph, features, id_map = self.preprocessor.test_data(df_test,
                                                              self.G_train)

        # Inference on test data
        predictions = self.graphsage_model.inference(
                [graph, features, id_map, self.walks, self.class_map],
                self.model_checkpoint)[1]

        # Compute predictions
        sorted_predictions = (-predictions).argsort(axis=1)
        conferenceseries = list()
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

        results = [conferenceseries, confidences]
        return results

    def train(self):
        pass

    def _load_training_graph(self):
        graph_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type, self.graph_type, "train_val-G.json")
        if os.path.isfile(graph_file):
            with open(graph_file) as f:
                self.G_train = json_graph.node_link_graph(json.load(f))
            return True
        return False

    def _load_training_class_map(self):
        class_map_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type, self.graph_type,
                "train_val-class_map.json")
        if isinstance(list(class_map.values())[0], list):
            lab_conversion = lambda n : n
        else:
            lab_conversion = lambda n : int(n)

        if os.path.isfile(class_map_file):
            self.class_map = json.load(open(class_map_file))
            self.class_map = {conversion(k): lab_conversion(v) for k, v in
                         class_map.items()}
            return True
        return False

    def _load_training_walks(self):
        walks_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type, self.graph_type, "train_val-walks.txt")
        self.walks = []
        if isinstance(list(self.G_train.nodes)[0], int):
            conversion = lambda n: int(n)
        else:
            conversion = lambda n: n

        if os.path.isfile(walks_file):
            with open(walks_file) as f:
                for line in f:
                    self.walks.append(map(conversion, line.split()))
            return True
        return False

    def _load_label_encoder(self):
        label_encoder_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type, self.graph_type, "label_encoder.pkl")
        if os.path.isfile(label_encoder_file):
            with open(label_encoder_file, "rb") as f:
                self.label_encoder = pickle.load(f)
                return True
        return False