# -*- coding: utf-8 -*-
import os
import sys
import pickle
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.join(os.getcwd(), "..", "data"))
from DataLoader import DataLoader
sys.path.insert(0, os.path.join(os.getcwd(), "..", "models"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "models", "authors_model"))
from AuthorsModel import AuthorsModel
sys.path.insert(0, os.path.join(os.getcwd(), "..", "models", "gat"))
from GATModel import GATModel
sys.path.insert(0, os.path.join(os.getcwd(), "..", "models", "han"))
from HANModel import HANModel


class ModelLoader():

    def __init__(self):
        self.models = []
        print("Preparing models.")
        authors_file = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), "data", "data_authors.pkl")
        with open(authors_file, "rb") as f:
            self.data_authors = pickle.load(f)
        self.model_authors = AuthorsModel()
        self.model_authors.train(self.data_authors)
        self.models.append("authors")

        self.model_gat = GATModel(
                embedding_type="SUM_2L", dataset="citations_authors_het_edges",
                graph_type="directed", hid_units=[64], n_heads=[8, 1],
                learning_rate=0.005, weight_decay=0, epochs=100000,
                batch_size=1, patience=100, residual=True,
                nonlinearity=tf.nn.elu, sparse=True, ffd_drop=0.5,
                attn_drop=0.5, gpu=0, recs=10, threshold=2)
        self.models.append("graph-attention-network")

        self.model_han = HANModel(
                model="HeteGAT_multi", embedding_type="AVG_L", hid_units=[128],
                n_heads=[8, 1], learning_rate=0.005, weight_decay=0,
                epochs=10000, batch_size=1, patience=100, residual=True,
                nonlinearity=tf.nn.elu, ffd_drop=0.5, attn_drop=0.5, gpu=0,
                recs=10)
        self.models.append("heterogeneous-graph-attention-network")

        data_file = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), "data", "data.pkl")
        with open(data_file, "rb") as f:
            self.data = pickle.load(f)

        # Load WikiCFP data
        wikicfp_file = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), "data", "wikicfp_data.pkl")
        with open(wikicfp_file, "rb") as f:
            self.wikicfp = pickle.load(f)
        print("Number of keys in wikicfp dictionary: ", len(self.wikicfp))

        # Load H5 Index rankings
        h5index_file = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), "data", "h5index_data.pkl")
        with open(h5index_file, "rb") as f:
            self.h5index = pickle.load(f)

        print("Model loader ready, models available")
        print(self.models)

    def get_models(self):
        return self.models

    def query_authors(self, model_name, data):
        print("Querying model: {}".format(model_name))
        names = list()
        for name in data:
            names.append(name.lower())
        recommendation = self.model_authors.query_single(names)
        return self._get_series_name(recommendation)

    def query_gnn(self, model_name, title, abstract, citations, authors):
        print("Querying model: {}".format(model_name))
        if model_name == "graph-attention-network":
            citations = self._get_citation_id(citations)
#            author_names = list()
#            for name in authors:
#                author_names.append(name.lower())
            recommendation = self.model_gat.query_single(
                    [title, abstract, citations, authors])
            return self._get_series_name(recommendation)
        elif model_name == "heterogeneous-graph-attention-network":
            citations = self._get_citation_id(citations)
            recommendation = self.model_han.query_single(
                    [title, abstract, citations, authors])
            return self._get_series_name(recommendation)
        else:
            print("Model not found. Please select a different model.")
            return False

    def _get_citation_id(self, citations):
        ids = ""
        for citation in citations:
            if citation is not "":
                citation_id = self.data[
                        self.data.chapter_title == citation].chapter.tolist()[
                                0]
                ids += citation_id + " "
        return ids

    def _get_series_name(self, recommendation):
        conferenceseries = list()
        confidence = list()
        wikicfp = list()
        h5index = list()
        for i, conf in enumerate(recommendation[0][0]):
            try:
                conferenceseries.append(
                        self.data[self.data.conferenceseries == conf].iloc[0][
                                  "conferenceseries_name"])
            except Exception as e:
                conferenceseries.append(conf)
            confidence.append(round(recommendation[1][0][i], 2))
            wikicfp.append(self._add_wikicfp(conf))
            h5index.append(self._add_h5index(conf))
        return [conferenceseries, confidence, wikicfp, h5index]

    def _add_wikicfp(self, conferenceseries):
        if conferenceseries in self.wikicfp:
            wikicfp = self.wikicfp[conferenceseries]
            # Limit description to 400 words
            if wikicfp["description"] is not None:
                wikicfp["description"] = wikicfp["description"][:400]
            return wikicfp
        else:
            return None

    def _add_h5index(self, conferenceseries):
        if conferenceseries in list(self.h5index.conferenceseries):
            h5index = self.h5index[
                    self.h5index.conferenceseries == conferenceseries
                    ].h5_index.tolist()
            if h5index is not None:
                return h5index[0]
            else:
                return None

    def autocomplete(self, model_name, data):
        if model_name == "authors":
            return self.model_authors.get_author_names(term=data)

    def autocomplete_authors(self, data):
        authors = pd.Series(self.data_authors.author_name.unique())
        authors = authors[authors.str.lower().str.startswith(
                data.lower())][:10]
        return authors

    def autocomplete_citations(self, data):
        citations = pd.Series(self.data["chapter_title"].unique())
        citations = citations[citations.str.lower().str.startswith(
                data.lower())][:10]
        return citations
