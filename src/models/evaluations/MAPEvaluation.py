# -*- coding: utf-8 -*-
from AbstractClasses import AbstractEvaluation


class MAPEvaluation(AbstractEvaluation):

    def evaluate(self, recommendation, truth):
        """Computes the mean average precision (MAP) for all queries.

        Args:
            recommendation (list): The list of recommendations returned
                                    by the model.
            truth (list): The test set list of conferences attended.

        Returns:
            int: the mean of all average precision scores for all the queries.
        """
        sum_avg_precisions = 0
        rank = 0

        # Sum the avearge precisions@k for all queries
        for i in range(len(recommendation[0])):
            sum_precisions = 0
            avg_precision = 0

            if truth[0][i] is not None:
                if recommendation[0][i] is not None:
                    checked = set()
                    for j in range(len(recommendation[0][i])):
                        if (recommendation[0][i][j] is not None) and (
                                recommendation[0][i][j] in truth[0][i]
                                ) and not(recommendation[0][i][j] in checked):
                            rank = j + 1
                            sum_precisions += self._precison_at_k(
                                    recommendation[0][i], truth[0][i], rank)
                            checked.add(recommendation[0][i][j])
                    avg_precision = sum_precisions/len(truth[0][i])
            sum_avg_precisions += avg_precision

        return sum_avg_precisions/len(recommendation[0])

    def _precison_at_k(self, recommendation, truth, k):
        count_relevant_retrieved = 0

        for i in range(len(recommendation)):
            rank = i + 1
            if (recommendation[i] in truth) and (rank <= k):
                count_relevant_retrieved += 1
        if k != 0:
            return count_relevant_retrieved/k
        else:
            return 0
