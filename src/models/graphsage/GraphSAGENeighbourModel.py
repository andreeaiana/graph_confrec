# -*- coding: utf-8 -*-
import os
import sys
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from networkx.readwrite import json_graph

sys.path.insert(0, os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "data"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "utils"))

from AbstractClasses import AbstractModel
from DataLoader import DataLoader
from preprocess_data import Processor
from similarities import Similarities
from unsupervised_model import UnsupervisedModel


class GraphSAGENeighbourModel(AbstractModel):

    def __init__(self, embedding_type, model_checkpoint, train_prefix,
                 model_name, model_size="small", learning_rate=0.00001,
                 epochs=10, dropout=0.0, weight_decay=0.0, max_degree=100,
                 samples_1=25, samples_2=10, dim_1=128, dim_2=128,
                 random_context=True, neg_sample_size=20, batch_size=512,
                 identity_dim=0, save_embeddings=False,
                 base_log_dir='../../../data/processed/graphsage/',
                 validate_iter=5000, validate_batch_size=256, gpu=0,
                 print_every=50, max_total_steps=10**10,
                 log_device_placement=False, recs=10):

        self.embedding_type = embedding_type
        self.model_checkpoint = model_checkpoint
        self.recs = recs
        self.gpu = gpu

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        self.graphsage_model = UnsupervisedModel(
                                train_prefix, model_name, model_size,
                                learning_rate, epochs, dropout, weight_decay,
                                max_degree, samples_1, samples_2, dim_1, dim_2,
                                random_context, neg_sample_size, batch_size,
                                identity_dim, save_embeddings, base_log_dir,
                                validate_iter, validate_batch_size, gpu,
                                print_every, max_total_steps,
                                log_device_placement)
        self.preprocessor = Processor(self.embedding_type, self.gpu)

        # Prepare the training data
        d_train = DataLoader()
        self.df_train = d_train.training_data_with_abstracts_citations().data

        print("Loading the training embeddings...")
        if not self._load_train_embeddings():
            print("The pretrained embeddings are missing.")
        else:
            print("Loaded.")

        training_ids = list(self.df_train.chapter)
        self.training_embeddings = self.pretrained_embeddings[[
                self.pretrained_embeddings_id_map[id] for id in training_ids]]
        self.sim = Similarities(self.training_embeddings, training_ids)

        print("Loading training graph...")
        if not self._load_training_graph():
            print("The training graph does not exist.")
        else:
            print("Loaded.")

        print("Loading training walks...")
        if not self._load_training_walks():
            print("The walks graph do not exist.")
        else:
            print("Loaded.")

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

        # Obtain the most similar neighbours
        similarities = []
        with tqdm(desc="Computing similarities",
                  total=len(test_embeddings)) as pbar:
            for vector in test_embeddings:
                similarities.append(self.sim.similar_by_vector(
                            vector, topn=self.recs*10))
                pbar.update(1)

        # Map similar papers to conferences
        conferenceseries = []
        confidences = []
        with tqdm(desc="Computing conference predicitons.",
                  total=len(similarities)) as pbar:
            for similarity in similarities:
                conferences = set()
                scores = []
                for idx in range(len(similarity)):
                    conferences_length = len(conferences)
                    if conferences_length < self.recs:
                        conferences.add(
                                list(self.df_train[
                                        self.df_train.chapter == similarity[
                                         idx][0]].conferenceseries)[0])
                        if len(conferences) != conferences_length:
                            scores.append(similarity[idx][1])
                conferenceseries.append(list(conferences))
                confidences.append(scores)
                pbar.update(1)

        results = [conferenceseries, confidences]
        return results

    def train(self):
        pass

    def _load_train_embeddings(self):
        embeddings_file = os.path.join(self.graphsage_model._log_dir(),
                                       "embeddings.npy")
        embeddings_ids_file = os.path.join(self.graphsage_model._log_dir(),
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

    def _load_training_graph(self):
        graph_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type, "train_val-G.json")
        if os.path.isfile(graph_file):
            with open(graph_file) as f:
                self.G_train = json_graph.node_link_graph(json.load(f))
            return True
        return False

    def _load_training_walks(self):
        walks_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "..", "data", "interim", "graphsage",
                self.embedding_type, "train_val-walks.txt")
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
