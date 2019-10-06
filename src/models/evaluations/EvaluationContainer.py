# -*- coding: utf-8 -*-
class EvaluationContainer():

    def __init__(self, evaluations=None):
        if evaluations is not None:
            self.evaluations = evaluations
        else:
            # Create a standard repertoire
            from MeanRecallEvaluation import MeanRecallEvaluation
            from MeanPrecisionEvaluation import MeanPrecisionEvaluation
            from MAPEvaluation import MAPEvaluation
            from MAPkEvaluation import MAPkEvaluation
            self.evaluations = {
                    "Recall": MeanRecallEvaluation(),
                    "Precision(1)": MeanPrecisionEvaluation(1),
                    "Precision(0)": MeanPrecisionEvaluation(0),
                    "MAP": MAPEvaluation(),
                    "MAP@1": MAPkEvaluation(1),
                    "MAP@3": MAPkEvaluation(3),
                    "MAP@5": MAPkEvaluation(5),
                    "MAP@10": MAPkEvaluation(10),
                    }
        self.max_len = str(len(max(self.evaluations, key=len)) + 1)

    def evaluate(self, recommendation, truth):
        results = []
        for evaluator_name, evaluator in self.evaluations.items():
            result = evaluator.evaluate(recommendation, truth)
            result_rounded = round(result, 3)
            results.append(result)
            print(("{:<" + self.max_len + "s}= {:0.3f}").format(
                    evaluator_name, result_rounded))

        print(" ".join([str(result) for result in results]))
        return results
