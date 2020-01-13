# -*- coding: utf-8 -*-
import os
import sys
import csv
import pickle
import argparse
import pandas as pd
from WikiCFPLinker import WikiCFPLinker


class WikiCFPLinkerEvaluation():

    def __init__(self, similarity_metric="damerau_levenshtein",
                 match_threshold=0.885, remove_stopwords=True):

        self.linker = WikiCFPLinker(similarity_metric, match_threshold,
                                    remove_stopwords)

        self.gold_standard_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..", "..", "data", "interim", "WikiCFP", "gold_standard.csv")

        self.persistent_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "data", "interim", "WikiCFP", "gold_standard.pkl")

    def evaluate(self):
        gold_standard = self._get_gold_standard()

        # Load the computed correspondences
        correspondences = self.linker.match_conferences()
        self.linker.get_statistics()

        # Evaluate the matching
        correct_predicted = 0
        for sg_series in gold_standard["scigraph_conferenceseries"]:
            if sg_series in list(correspondences["conferenceseries"]):
                predicted = correspondences[
                        correspondences["conferenceseries"] == sg_series][
                                "WikiCFP_conferenceseries"].tolist()[0]
                truth = gold_standard[gold_standard[
                        "scigraph_conferenceseries"] == sg_series][
                        "wikicfp_conferenceseries"].tolist()[0]
                if predicted == truth:
                    correct_predicted += 1

        recall = correct_predicted/len(gold_standard)
        precision = correct_predicted/len(correspondences)

        if recall != 0 and precision != 0:
            f1_measure = 2*precision*recall/(precision+recall)
        else:
            f1_measure = 0
        print("Precision: {}, Recall: {}, F1-Measure: {}.".format(
                precision, recall, f1_measure))

    def _get_gold_standard(self):
        """Reads, processes, and saves the Gold Standard .csv file if not
        already processed and pickled.
        """
        if not self._load_gold_standard():
            gold_std = list()
            if os.path.isfile(self.gold_standard_file):
                print("Reading and processing gold standard.")
                with open(self.gold_standard_file, newline='') as f:
                    reader = csv.DictReader(f)
                    try:
                        for row in reader:
                            gold_std.append((row["conferenceseries"],
                                             row["wikicfp_name"]))
                    except csv.Error as e:
                        sys.exit('file {}, line {}: {}'.format(
                                self.gold_standard_file, reader.line_num, e))

            self.gold_standard = pd.DataFrame(
                    gold_std,
                    columns=["scigraph_conferenceseries",
                             "wikicfp_conferenceseries"])
            self._saveGoldStandard()
        return self.gold_standard

    def _load_gold_standard(self):
        """Loads the processed Gold Standard.
        """
        if os.path.isfile(self.persistent_file):
            print("Loading gold standard.")
            with open(self.persistent_file, "rb") as f:
                self.gold_standard = pickle.load(f)
                print("Loaded.")
                return True
        return False

    def _save_gold_standard(self):
        """Saves the processed Gold Standard as a pickle file.
        """
        print("Saving the gold standard to disk.")
        with open(self.persistent_file, "wb") as f:
            pickle.dump(self.gold_standard, f)

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for WikiCFPLinker.')
        parser.add_argument('--similarity_metric',
                            choices=["levenshtein", "damerau_levenshtein",
                                     "jaro", "jaro_winkler"],
                            default="damerau_levenshtein",
                            help="Type of similarity metric used.")
        parser.add_argument('--match_threshold',
                            default=0.885,
                            help='The matching threshold.')
        parser.add_argument('--remove_stopwords',
                            default=True,
                            action="store_false",
                            help='The type of graph used ' +
                            '(directed vs. undirected).')
        args = parser.parse_args()
        print("Starting...")
        from WikiCFPLinkerEvaluation import WikiCFPLinkerEvaluation
        evaluator = WikiCFPLinkerEvaluation(
                args.similarity_metric, args.match_threshold,
                args.remove_stopwords)
        evaluator.evaluate()
        print("Finished.")

    if __name__ == "__main__":
        main()
