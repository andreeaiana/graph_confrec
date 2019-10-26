# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), ".."))
from AbstractClasses import AbstractEvaluation


class MeanPrecisionEvaluation(AbstractEvaluation):

    def __init__(self, none_value=1):
        self.none_value = none_value

    def evaluate(self, recommendation, truth):
        """Calculates the mean precision (i.e. the fraction of attended
        conferences that were recommended).

        Args:
            recommendation (list): The list of recommendations returned
                                    by the model.
            truth (list): The test set list of conferences attended.

        Returns:
            int: #recommended and attended conferences / #attended conferences
        """
        precision = 0
        size = len(recommendation[0])
        for i in range(size):
            if recommendation[0][i] is None:
                precision += self.none_value
            elif truth[0][i] is None:
                precision += 0
            else:
                recommendation_set = set(recommendation[0][i])
                truth_set = set(truth[0][i])
                precision += len(recommendation_set.intersection(truth_set)
                                 ) / len(recommendation_set)

        return precision / size
