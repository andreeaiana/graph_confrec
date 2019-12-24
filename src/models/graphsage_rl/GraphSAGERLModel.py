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
from supervised_model import SupervisedModelRL


class GraphSAGERLModel(AbstractModel):

    def __init__(self, embedding_type, graph_type, train_prefix, model_name,
                 nonlinear_sampler=True, fast_ver=False, allhop_rewards=False,
                 model_size="small", learning_rate=0.001, epochs=10,
                 dropout=0.0, weight_decay=0.0, max_degree=100, samples_1=25,
                 samples_2=10, samples_3=0, dim_1=512, dim_2=512, dim_3=0,
                 batch_size=128, sigmoid=False, identity_dim=0,
                 base_log_dir='../../../data/processed/graphsage_rl/',
                 validate_iter=5000, validate_batch_size=128, gpu=0,
                 print_every=5, max_total_steps=10**10,
                 log_device_placement=False, recs=10, threshold=2):

        self.embedding_type = embedding_type
        self.graph_type = graph_type
        self.fast_ver = fast_ver
        self.recs = recs

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        self.graphsage_model = SupervisedModelRL(
                train_prefix, model_name, nonlinear_sampler, fast_ver,
                allhop_rewards, model_size, learning_rate, epochs, dropout,
                weight_decay, max_degree, samples_1, samples_2, samples_3,
                dim_1, dim_2, dim_3, batch_size, sigmoid, identity_dim,
                base_log_dir, validate_iter, validate_batch_size, gpu,
                print_every, max_total_steps, log_device_placement)
        self.preprocessor = Processor(self.embedding_type, self.graph_type,
                                      threshold, gpu)

        if not self._load_training_graph():
            print("The training graph does not exist.")

        if not self._load_training_class_map():
            print("The training class map dows not exist.")

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

        if self.graph_type == "citations":
            if len(query) < 3:
                raise ValueError("The input does not contain enough data; " +
                                 "chapter  title chapter abstract, and " +
                                 "chapter citations are required.")
            return self.query_batch([(query_id, query[0], query[1], query[2])])
        elif self.graph_type == "citations_authors_het_edges":
            if len(query) < 4:
                raise ValueError("The input does not contain enough data; " +
                                 "chapter title chapter abstract, chapter " +
                                 "citations, and chapter authors are required."
                                 )
            authors_df = pd.DataFrame({"author_name": query[3],
                                       "chapter": [query_id]*len(query[3])})
            return self.query_batch([(query_id, query[0], query[1], query[2])],
                                    authors_df)
        else:
            raise ValueError("Graph type not recognised.")


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
        if self.graph_type == "citations":
            df_test = pd.DataFrame(batch, columns=["chapter", "chapter_title",
                                                   "chapter_abstract",
                                                   "chapter_citations"])

            # Preprocess the data
            graph, features, id_map, class_map = self.preprocessor.test_data(
                    df_test, self.G_train, class_map=self.class_map_train)

        elif self.graph_type == "citations_authors_het_edges":
            df_test = pd.DataFrame(batch[0],
                                   columns=["chapter", "chapter_title",
                                            "chapter_abstract",
                                            "chapter_citations"])
            authors_df = batch[1]
            # Preprocess the data
            graph, features, id_map, class_map = self.preprocessor.test_data(
                    df_test, self.G_train, authors_df=authors_df,
                    class_map=self.class_map_train)
        else:
            raise ValueError("Graph type not recognised.")

        # Inference on test data
        if self.fast_ver:
            sampler_name = "FastML"
        else:
            sampler_name = "ML"
        predictions = self.graphsage_model.inference(
                [graph, features, id_map, None, class_map], sampler_name)[1]

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

    def _load_training_graph(self):
        graph_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type, self.graph_type, "train_val-G.json")
        if os.path.isfile(graph_file):
            print("Loading training graph...")
            with open(graph_file) as f:
                self.G_train = json_graph.node_link_graph(json.load(f))
            print("Loaded.")
            return True
        return False

    def _load_training_class_map(self):
        class_map_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type, self.graph_type,
                "train_val-class_map.json")
        self.class_map_train = {}
        if isinstance(list(self.G_train.nodes)[0], int):
            conversion = lambda n: int(n)
        else:
            conversion = lambda n: n
        if os.path.isfile(class_map_file):
            print("Loading training class map...")
            self.class_map_train = json.load(open(class_map_file))
            if isinstance(list(self.class_map_train.values())[0], list):
                lab_conversion = lambda n : n
            else:
                lab_conversion = lambda n : int(n)
            self.class_map_train = {conversion(k): lab_conversion(v) for k, v
                                    in self.class_map_train.items()}
            print("Loaded.")
            return True
        return False

    def _load_label_encoder(self):
        label_encoder_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type, self.graph_type, "label_encoder.pkl")
        if os.path.isfile(label_encoder_file):
            with open(label_encoder_file, "rb") as f:
                print("Loading label encoder.")
                self.label_encoder = pickle.load(f)
            print("Loaded.")
            return True
        return False
