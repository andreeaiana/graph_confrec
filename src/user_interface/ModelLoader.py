# -*- coding: utf-8 -*-
import os
import sys
import pickle
import pandas as pd

sys.path.insert(0, os.path.join(os.getcwd(), "..", "data"))
from DataLoader import DataLoader
sys.path.insert(0, os.path.join(os.getcwd(), "..", "models"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "models", "authors_model"))
from AuthorsModel import AuthorsModel
sys.path.insert(0, os.path.join(os.getcwd(), "..", "models", "gat"))
from GATModel import GATModel


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

    def query(self, model_name, data):
        print("Querying model: {}".format(model_name))
        if model_name == "authors":
            names = list()
            for name in data:
                names.append(name.lower())
            recommendation = self.model_authors.query_single(names)
            return self._get_series_name(recommendation)
        else:
            print("Model not found. Please select a different model.")
            return False

    def _get_series_name(self, recommendation):
        conferenceseries = list()
        confidence = list()
        wikicfp = list()
        h5index = list()
        for i, conf in enumerate(recommendation[0][0]):
            conferenceseries.append(
                    self.data[self.data.conferenceseries == conf].iloc[0][
                              "conferenceseries_name"])
            confidence.append(round(recommendation[1][0][i], 2))
            wikicfp.append(self._add_wikicfp(conf))
            h5index.append(self._add_h5index(conf))
        print([conferenceseries, confidence, wikicfp, h5index])
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

#
#    def __init__(self):
#
#        d = DataLoader()
#        d.papers(["2013","2014","2015", "2016"]).conferences().conferenceseries().keywords()
#        self.data_tags = d.data.loc[:, ["keyword", "keyword_label"]]
#        #d = DataLoader()
#        #d.training_data_for_abstracts("small")
#        #self.data_abstracts = d.data.copy()
#        file = os.path.join(".", "data", "data_abstracts.pkl")
#        #with open(file,"wb") as f:
#        #    pickle.dump(self.data_abstracts, f)
#        with open(file,"rb") as f:
#                self.data_abstracts = pickle.load(f)
#        #del d
#        self.model_tfidf_union = TfIdfUnionAbstractsModel(ngram_range=(1,4), max_features=1000000)
#        self.model_tfidf_union.train(data=self.data_abstracts, data_name="small")
#        self.models.append("nTfIdf_concat")
#        self.model_doc2vec = Doc2VecUnionAbstractsModel(embedding_model="d2v_100d_w5_NS")
#        self.model_doc2vec.train(data=self.data_abstracts, data_name="small")
#        self.models.append("Doc2Vec")
#        self.model_cnn = CNNAbstractsModel(net_name="CNN2-100f-2fc-0.0005decay")
#        self.models.append("CNN")
#        self.model_keyword = KeywordsUnionAbstractsModel()
#        self.model_keyword._load_model("small")
#        self.models.append("Keywords_TfIdf")
#        ensemble_tfidf = TfIdfUnionAbstractsModel(ngram_range=(1,4), max_features=1000000, recs=100)
#        ensemble_tfidf.train(data=self.data_abstracts, data_name="small")
#        ensemble_cnn = CNNAbstractsModel(net_name="CNN2-100f-2fc-0.0005decay", recs=100)
#        ensemble_keyword = KeywordsUnionAbstractsModel(recs=100)
#        ensemble_keyword._load_model("small")
#        self.model_ensemble = EnsembleStackModel(
#            models=[
#                    ensemble_tfidf
#                    ,ensemble_cnn
#                    ,ensemble_keyword
#            ],
#            is_abstract=[
#                    True
#                    ,True
#                    ,False
#            ],
#            max_recs_models=100
#        )
#        self.model_ensemble._load_model("small")
#        self.models.append("Ensemble")
#
#
#    def query(self,modelName, data):
#        print("querying model: " + modelName)
#        if modelName=="Authors":
#            names = list()
#            for d in data:
#                names.append(d.lower())
#            rec = self.model_authors.query_single(names)
#            return self.addDummyConfidence(rec)
#        if modelName=="nTfIdf_concat":
#            rec = self.model_tfidf_union.query_single(data)
#            return self.getSeriesNames(rec)
#        if modelName=="Doc2Vec":
#            rec = self.model_doc2vec.query_batch(data)
#            return self.getSeriesNames(rec)
#        if modelName=="CNN":
#            rec = self.model_cnn.query_single(data)
#            return self.getSeriesNames(rec)
#        if modelName=="Keywords_TfIdf":
#            rec = self.model_keyword.query_single(self.getKeywordIDs(data))
#            return self.getSeriesNames(rec)
#        print("Model not found, please select a different model")
#        return False
#
#    def query_ensemble(self, abstract, keywords):
#        print("querying ensemble model")
#        keys = self.getKeywordIDs(keywords)
#        rec = self.model_ensemble.query_single(abstract, keys)
#        return self.getSeriesNames(rec)

#    def autocomplete(self, modelName, data):
#        if modelName == "Authors":
#            return self.model_authors.get_author_names(term=data)
#        if modelName == "Keywords_TfIdf" or modelName=="Ensemble":
#            tags = pd.Series(self.data_tags["keyword_label"].unique())
#
#            tags = tags[tags.str.lower().str.startswith(data.lower())][:10]
#
#            return tags
#
#
#
#    def getKeywordIDs(self, data):
#        ids = ""
#        for d in data:
#            if d is not "":
#                tmp = self.data_tags[self.data_tags.keyword_label == d].iloc[0].keyword
#                tmp = tmp.replace("<http://scigraph.springernature.com/things/product-market-codes/","")
#                tmp = tmp[0:-1]
#                ids += tmp + " "
#        return ids