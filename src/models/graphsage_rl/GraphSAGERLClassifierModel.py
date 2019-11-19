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
from unsupervised_model import UnsupervisedModelRL


class GraphSAGERLClassifierModel(AbstractModel):

    def __init__(self, classifier, embedding_type, graph_type,
                 model_checkpoint, train_prefix, model_name,
                 sampler_name="FastML", nonlinear_sampler=False,
                 uniform_ratio=0.6, model_size="small", learning_rate=0.00001,
                 epochs=10, dropout=0.0, weight_decay=0.0, max_degree=100,
                 samples_1=25, samples_2=10, dim_1=128, dim_2=128,
                 random_context=True, neg_sample_size=20, batch_size=512,
                 identity_dim=0, save_embeddings=False,
                 base_log_dir='../../../data/processed/graphsage_rl/',
                 validate_iter=5000, validate_batch_size=512, gpu=0,
                 print_every=50, max_total_steps=10**10,
                 log_device_placement=False, recs=10):

        self.classifier = classifier
        self.embedding_type = embedding_type
        self.graph_type = graph_type
        self.sampler_name = sampler_name
        self.model_checkpoint = model_checkpoint
        self.recs = recs

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        self.graphsage_model = UnsupervisedModelRL(
                                train_prefix, model_name, nonlinear_sampler,
                                uniform_ratio, model_size, learning_rate,
                                epochs, dropout, weight_decay, max_degree,
                                samples_1, samples_2, dim_1, dim_2,
                                random_context, neg_sample_size, batch_size,
                                identity_dim, save_embeddings, base_log_dir,
                                validate_iter, validate_batch_size, gpu,
                                print_every, max_total_steps,
                                log_device_placement)
        self.preprocessor = Processor(self.embedding_type, "citations", gpu)

        self.classifier_file = os.path.join(
                self.graphsage_model._log_dir(self.sampler_name),
                self.classifier.__class__.__name__ + ".pkl")

        if not self._load_training_graph():
            print("The training graph does not exist.")

        if not self._load_training_walks():
            print("The walks do not exist.")

    def query_single(self, query):
        """Queries the model and returns a list of recommendations.

        Args:
            query (list): The query as needed by the model, is in the form
            [chapter_title, chapter_abstract, list(chapter_citations)].

        Returns
            list: ids of the conferences
            double: confidence scores
        """
        if len(query) < 3:
            print("The input does not contain enough data; chapter title " +
                  "chapter abstract, and chapter citations are required.")
        # Generate an ID for the query
        query_id = "new_node_id:" + "-".join(
                [str(i) for i in random.sample(range(0, 10000), 5)])
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

        # Infer embeddings
        test_nodes, test_embeddings = self.graphsage_model.predict(
                [graph, features, id_map, self.walks],
                self.model_checkpoint)

        # Compute predictions
        predictions = self.classifier.predict_proba(test_embeddings)
        sorted_predictions = np.argsort(-np.array(predictions))

        conferenceseries = list()
        confidences = list()

        for index, order in enumerate(sorted_predictions):
            conferences = list()
            scores = list()
            i = 0
            while len(conferences) < self.recs:
                conf = self.label_encoder.inverse_transform(
                        [order[i]]).tolist()[0]
                if conf not in conferences:
                    conferences.append(conf)
                    scores.append(predictions[index][order][i])
                i += 1
            conferenceseries.append(conferences)
            confidences.append(scores)

        results = [conferenceseries, confidences]
        return results

    def train(self, data):
        if not self._load_model_classifier():
            print("Classifier not trained yet. Training now...")
            timer = Timer()
            timer.tic()

            print("Loading the training embeddings...")
            if not self._load_train_embeddings():
                print("The pretrained embeddings are missing.")
            else:
                print("Loaded.")
            training_ids = list(data.chapter)
            training_embeddings = self.pretrained_embeddings[[
                    self.pretrained_embeddings_id_map[id] for id in
                    training_ids]]

            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(
                    data.conferenceseries)
            self.classifier.fit(training_embeddings, self.labels)
            self._save_model_classifier()

            print("Training finished.")
            timer.toc()

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
            print("Loading training walks...")
            with open(walks_file) as f:
                for line in f:
                    self.walks.append(map(conversion, line.split()))
            print("Loaded.")
            return True
        return False

    def _load_train_embeddings(self):
        embeddings_file = os.path.join(
                self.graphsage_model._log_dir(self.sampler_name),
                "embeddings.npy")
        embeddings_ids_file = os.path.join(
                self.graphsage_model._log_dir(self.sampler_name),
                "embeddings_ids.txt")
        if os.path.isfile(embeddings_file) and os.path.isfile(
                embeddings_ids_file):
            self.pretrained_embeddings = np.load(embeddings_file)
            self.pretrained_embeddings_id_map = {}
            with open(embeddings_ids_file) as f:
                for i, line in enumerate(f):
                    self.pretrained_embeddings_id_map[line.strip()] = i
            return True
        return False

    def _load_model_classifier(self):
        if os.path.isfile(self.classifier_file):
            print("Loading classifier...")
            with open(self.classifier_file, "rb") as f:
                self.label_encoder, self.labels, self.classifier = pickle.load(
                        f)
                print("Loaded.")
                return True
        return False

    def _save_model_classifier(self):
        with open(self.classifier_file, "wb") as f:
            pickle.dump([self.label_encoder, self.labels, self.classifier],
                        f, protocol=4)

    def _has_persistent_model(self):
        if os.path.isfile(self.classifier_file):
            return True
        return False
