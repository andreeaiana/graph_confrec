# -*- coding: utf-8 -*-
from AbstractClasses import AbstractEvaluation


class MAPkEvaluation(AbstractEvaluation):

    def __init__(self, k=10, duplicates=True):
        self.k = k
        if duplicates:
            self.evaluate = self._evaluate_duplicates
        else:
            self.evaluate = self._evaluate_no_duplicates

    def _evaluate_no_duplicates(self, recommendation, truth):
        map_sum = 0
        for i, q in enumerate(recommendation[0]):
            hits = 0
            ap = 0
            try:
                for rank, rec in enumerate(q[0:self.k], 1):
                    add = rec in truth[0][i]
                    hits += add
                    ap += add*(hits/rank)
                map_sum += ap/max(len(truth[0][i]), self.k)
            except TypeError:
                # equivalent to adding 0 to map_sum if recommendation is None
                pass
        return map_sum/len(recommendation[0])

    def _evaluate_duplicates(self, recommendation, truth):
        map_sum = 0
        for i, q in enumerate(recommendation[0]):
            hits = 0
            ap = 0
            hitlist = []
            try:
                for rank, rec in enumerate(q[0:self.k], 1):
                    add = rec in truth[0][i]
                    hits += add
                    add = add & (rec not in hitlist)
                    if add:
                        hitlist.append(rec)
                    ap += add*(hits/rank)
                map_sum += ap/min(len(truth[0][i]), self.k)
            except TypeError:
                # equivalent to adding 0 to map_sum if recommendation is None
                pass
        return map_sum/len(recommendation[0])
