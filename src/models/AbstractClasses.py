# -*- coding: utf-8 -*-
import pandas as pd


class AbstractModel:

    def query_single(self, query):
        """Queries the model and returns a list of recommendations.

        Args:
            query (list): The query as needed by the model.

        Returns
            list: ids of the conferences
            double: confidence scores
        """
        pass

    def query_batch(self, batch):
        """Queries the model and returns a lis of recommendations.

        Args:
            batch (list): The list of queries as needed by the model.

        Returns
            list: ids of the conferences
            double: confidence scores
        """
        pass

    def train(self, data):
        """Set the data to train the model. Will fail if "data" is not a
        pandas DataFrame.

        Args:
            data (pandas.DataFrame): The data used by the model.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Argument 'data' needs to be of type \
                            pandas.DataFrame.")
        pass


class AbstractEvaluation:
    def evaluate(self, recommendation, truth):
        return
