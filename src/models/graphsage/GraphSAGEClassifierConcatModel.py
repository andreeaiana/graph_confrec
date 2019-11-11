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
from unsupervised_model import UnsupervisedModel


class GraphSAGEClassifierConcatModel(AbstractModel):

    def __init__(self, classifier, embedding_type, model_checkpoint_citations,
                 model_checkpoint_authors, train_prefix_citations,
                 train_prefix_authors, model_name, model_size="small",
                 learning_rate=0.00001, epochs=10, dropout=0.0,
                 weight_decay=0.0, max_degree=100, samples_1=25, samples_2=10,
                 dim_1=128, dim_2=128, random_context=True, neg_sample_size=20,
                 batch_size=512, identity_dim=0, save_embeddings=False,
                 base_log_dir='../../../data/processed/graphsage/',
                 validate_iter=5000, validate_batch_size=256, gpu=0,
                 print_every=50, max_total_steps=10**10,
                 log_device_placement=False, recs=10):

        self.classifier = classifier
        self.embedding_type = embedding_type
        self.model_checkpoint_citations = model_checkpoint_citations
        self.model_checkpoint_authors = model_checkpoint_authors
        self.recs = recs

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # GraphSAGE models
        self.graphsage_model_citations = UnsupervisedModel(
                                train_prefix_citations, model_name, model_size,
                                learning_rate, epochs, dropout, weight_decay,
                                max_degree, samples_1, samples_2, dim_1, dim_2,
                                random_context, neg_sample_size, batch_size,
                                identity_dim, save_embeddings, base_log_dir,
                                validate_iter, validate_batch_size, gpu,
                                print_every, max_total_steps,
                                log_device_placement)
        self.graphsage_model_authors = UnsupervisedModel(
                                train_prefix_authors, model_name, model_size,
                                learning_rate, epochs, dropout, weight_decay,
                                max_degree, samples_1, samples_2, dim_1, dim_2,
                                random_context, neg_sample_size, batch_size,
                                identity_dim, save_embeddings, base_log_dir,
                                validate_iter, validate_batch_size, gpu,
                                print_every, max_total_steps,
                                log_device_placement)

        # Data preprocessors
        self.preprocessor_citations = Processor(self.embedding_type,
                                                "citations", gpu)
        self.preprocessor_authors = Processor(self.embedding_type,
                                                "authors", gpu)

        # Classifier file location
        classifier_dir = os.path.join(os.path.dirname(
                         os.path.realpath(__file__)), "..", "..", "..", "data",
                         "processed", "graphsage", self.embedding_type,
                         "citations_authors")
        classifier_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
                model=model_name, model_size=model_size, lr=learning_rate)
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)
        self.classifier_file = os.path.join(classifier_dir,
                                            self.classifier.__class__.__name__
                                            + ".pkl")

        # Load training graphs
        if not self._load_training_graph_citations():
            print("The citations training graph does not exist.")
        if not self._load_training_graph_authors():
            print("The authors training graph does not exist.")

        # Load training walks
        if not self._load_training_walks_citations():
            print("The citations walks do not exist.")
        if not self._load_training_walks_authors():
            print("The authors walks do not exist.")

    def query_single(self, query):
        """Queries the model and returns a list of recommendations.

        Args:
            query (list): The query as needed by the model, is in the form
            [chapter_title, chapter_abstract, list(chapter_citations),
            list(chapter_authors)].

        Returns
            list: ids of the conferences
            double: confidence scores
        """
        if len(query) < 4:
            print("The input does not contain enough data; chapter title " +
                  "chapter abstract, chapter citations, and chpater authors " +
                  "are required.")
        # Generate an ID for the query
        query_id = "new_node_id:" + "-".join(
                [str(i) for i in random.sample(range(0, 10000), 5)])
        authors_df = pd.DataFrame({"author_name": query[3],
                                   "chapter": [query_id]*len(query[3])})
        return self.query_batch([(query_id, query[0], query[1], query[2])],
                                authors_df)

    def query_batch(self, batch):
        """Queries the model and returns a lis of recommendations.

        Args:
            batch (tuple): A tuple containing the list of queries as needed by
            the model, in the form (chapter_id, chapter_title,
            chapter_abstract, list(chapter_citations)), and a dataframe of
            authors, in the form (author_name, chapter).

        Returns
            list: ids of the conferences
            double: confidence scores
        """

        df_test = pd.DataFrame(batch[0], columns=["chapter", "chapter_title",
                                                  "chapter_abstract",
                                                  "chapter_citations"])
        authors_df = batch[1]

        # Preprocess the data
        graph_citations, features_citations, id_map_citations = self.preprocessor.test_data(
                df_test, self.G_train_citations)
        graph_authors, features_authors, id_map_authors = self.preprocessor.test_data(
                df_test, self.G_train_authors, authors_df = authors_df)

        # Infer embeddings
        _, test_embeddings_citations = self.graphsage_model_citations.predict(
                [graph_citations, features_citations, id_map_citations,
                 self.walks_citations], self.model_checkpoint_citations)
        _, test_embeddings_authors = self.graphsage_model_authors.predict(
                [graph_authors, features_authors, id_map_authors,
                 self.walks_authors], self.model_checkpoint_authors)

        # Concatenate embeddings
        test_embeddings = np.concatenate((test_embeddings_citations,
                                          test_embeddings_authors), axis=1)

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

            # Load trained eembeddings
            print("Loading the citations training embeddings...")
            if not self._load_train_embeddings_citations():
                print("The pretrained citations embeddings are missing.")
            else:
                print("Loaded.")

            print("Loading the authors training embeddings...")
            if not self._load_train_embeddings_authors():
                print("The pretrained authors embeddings are missing.")
            else:
                print("Loaded.")

            training_ids = list(data.chapter)
            training_embeddings_citations = self.pretrained_embeddings_citations[[
                    self.pretrained_embeddings_id_map_citations[id] for id in
                    training_ids]]
            training_embeddings_authors = self.pretrained_embeddings_authors[[
                    self.pretrained_embeddings_id_map_authors[id] for id in
                    training_ids]]

            # Concatenate embeddings
            training_embeddings = np.concatenate((
                    training_embeddings_citations, training_embeddings_authors
                    ), axis=1)

            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(
                    data.conferenceseries)
            self.classifier.fit(training_embeddings, self.labels)
            self._save_model_classifier()

            print("Training finished.")
            timer.toc()

    def _load_training_graph_citations(self):
        graph_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type, "citations/train_val-G.json")
        if os.path.isfile(graph_file):
            print("Loading training graph...")
            with open(graph_file) as f:
                self.G_train_citations = json_graph.node_link_graph(
                        json.load(f))
            print("Loaded.")
            return True
        return False

    def _load_training_graph_authors(self):
        graph_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type, "authors/train_val-G.json")
        if os.path.isfile(graph_file):
            print("Loading training graph...")
            with open(graph_file) as f:
                self.G_train_authors = json_graph.node_link_graph(
                        json.load(f))
            print("Loaded.")
            return True
        return False

    def _load_training_walks_citations(self):
        walks_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type, "citations/train_val-walks.txt")
        self.walks_citations = []
        if isinstance(list(self.G_train_citations.nodes)[0], int):
            conversion = lambda n: int(n)
        else:
            conversion = lambda n: n

        if os.path.isfile(walks_file):
            print("Loading training walks...")
            with open(walks_file) as f:
                for line in f:
                    self.walks_citations.append(map(conversion, line.split()))
            print("Loaded.")
            return True
        return False

    def _load_training_walks_authors(self):
        walks_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type, "authors/train_val-walks.txt")
        self.walks_authors = []
        if isinstance(list(self.G_train_authors.nodes)[0], int):
            conversion = lambda n: int(n)
        else:
            conversion = lambda n: n

        if os.path.isfile(walks_file):
            print("Loading training walks...")
            with open(walks_file) as f:
                for line in f:
                    self.walks_authors.append(map(conversion, line.split()))
            print("Loaded.")
            return True
        return False

    def _load_train_embeddings_citations(self):
        embeddings_file = os.path.join(
                self.graphsage_model_citations._log_dir(), "embeddings.npy")
        embeddings_ids_file = os.path.join(
                self.graphsage_model_citations._log_dir(),
                "embeddings_ids.txt")
        if os.path.isfile(embeddings_file) and os.path.isfile(
                embeddings_ids_file):
            self.pretrained_embeddings_citations = np.load(embeddings_file)
            self.pretrained_embeddings_id_map_citations = {}
            with open(embeddings_ids_file) as f:
                for i, line in enumerate(f):
                    self.pretrained_embeddings_id_map_citations[
                            line.strip()] = i
            return True
        return False

    def _load_train_embeddings_authors(self):
        embeddings_file = os.path.join(
                self.graphsage_model_authors._log_dir(), "embeddings.npy")
        embeddings_ids_file = os.path.join(
                self.graphsage_model_authors._log_dir(),
                "embeddings_ids.txt")
        if os.path.isfile(embeddings_file) and os.path.isfile(
                embeddings_ids_file):
            self.pretrained_embeddings_authors = np.load(embeddings_file)
            self.pretrained_embeddings_id_map_authors = {}
            with open(embeddings_ids_file) as f:
                for i, line in enumerate(f):
                    self.pretrained_embeddings_id_map_authors[
                            line.strip()] = i
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
