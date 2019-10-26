# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), ".."))
from AbstractClasses import AbstractEvaluation


class MeanRecallEvaluation(AbstractEvaluation):

    def evaluate(self, recommendation, truth):
        """Calculates the mean recall (i.e. the fraction of attended
        conferences that were recommended).

        Args:
            recommendation (list): The list of recommendations returned
                                    by the model.
            truth (list): The test set list of conferences attended.

        Returns:
            int: #recommended and attended conferences / #attended conferences
        """
        recall = 0
        size = len(recommendation[0])
        for i in range(size):
            if truth[0][i] is None:
                recall += 1
            elif recommendation[0][i] is None:
                recall += 0
            else:
                recommendation_set = set(recommendation[0][i])
                truth_set = set(truth[0][i])
                recall += len(recommendation_set.intersection(truth_set)
                              ) / len(truth_set)

        return recall/size
