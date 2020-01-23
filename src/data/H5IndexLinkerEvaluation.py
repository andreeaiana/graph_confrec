# -*- coding: utf-8 -*-
import os
import sys
import pickle
import argparse
import pandas as pd

from H5IndexLinker import H5IndexLinker


class H5IndexLinkerEvaluation:

    def __init__(self, similarity_metric="damerau_levenshtein",
                 threshold=0.9):
        self.gold_standard_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "data", "interim", "H5Index")
        self.linker = H5IndexLinker(similarity_metric, threshold)

    def evaluate(self):
        # load gold standard
        self._get_gold_standard()

        # load correspondences
        self.linker._load_correspondences()
        correspondences = self.linker.correspondences

        # Evaluate matching
        print("Evaluating...")
        correct_predicted = 0
        for conf in self.gold_standard.conferenceseries:
            if conf in list(correspondences.conferenceseries):
                predicted = correspondences[
                        correspondences.conferenceseries == conf
                        ].publication.tolist()[0]
                truth = self.gold_standard[
                        self.gold_standard.conferenceseries == conf
                        ].publication.tolist()[0]
                if predicted == truth:
                    correct_predicted += 1

        recall = correct_predicted / len(self.gold_standard)
        precision = correct_predicted / len(correspondences)

        if recall != 0 and precision != 0:
            f1_measure = 2 * precision * recall / (precision + recall)
        else:
            f1_measure = 0
        print("Evaluated.")
        print("Precision: {}, Recall: {}, F1-Measure: {}".format(
                precision, recall, f1_measure))
        self.linker.get_statistics()

    def _get_gold_standard(self):
        if not self._load_gold_standard():
            print("Gold standard not processed yet. Processing now...")
            file = os.path.join(self.gold_standard_path, "gold_standard.csv")
            self.gold_standard = pd.read_csv(file)
            self._save_gold_standard()
            print("Processed.")

    def _load_gold_standard(self):
        file = os.path.join(self.gold_standard_path, "gold_standard.pkl")
        if os.path.isfile(file):
            print("Loading gold standard")
            with open(file, "rb") as f:
                self.gold_standard = pickle.load(f)
                print("Loaded.")
                return True
        return False

    def _save_gold_standard(self):
        file = os.path.join(self.gold_standard_path, "gold_standard.pkl")
        print("Saving gold standard to disk.")
        with open(file, "wb") as f:
            pickle.dump(self.gold_standard, f)
        print("Saved.")

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for H5IndexLinkerEvaluation.')
        parser.add_argument('--similarity_metric',
                            choices=["levenshtein", "damerau_levenshtein",
                                     "jaro", "jaro_winkler"],
                            default="damerau_levenshtein",
                            help="Type of similarity metric used.")
        parser.add_argument('--threshold',
                            default=0.9,
                            help='The matching threshold.')
        args = parser.parse_args()
        print("Starting...")
        from H5IndexLinkerEvaluation import H5IndexLinkerEvaluation
        linker = H5IndexLinkerEvaluation(args.similarity_metric,
                                         args.threshold)
        linker.evaluate()
        print("Finished.")

    if __name__ == "__main__":
        main()
