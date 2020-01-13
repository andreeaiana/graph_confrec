# -*- coding: utf-8 -*-
import os
import sys
import csv
import pickle
import pandas as pd


class WikiCFPLinkerGoldStandard():

    def __init__(self):
        self.gold_standard_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..", "..", "data", "interim", "WikiCFP", "gold_standard.csv")
        self.persistent_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "..", "data", "interim", "WikiCFP", "gold_standard.pkl")

    def get_gold_standard(self):
        """
        Reads, processes, and saves the Gold Standard .csv file if not already
        processed and pickled.
        """
        if not self._load_gold_standard():
            gold_std = list()
            if os.path.isfile(self.gold_standard_file):
                print("Reading and processing gold standard.")
                with open(self.gold_standard_file, newline='') as f:
                    reader = csv.DictReader(f)
                    try:
                        for row in reader:
                            gold_std.append((
                                    row["conferenceseries"],
                                    row["WikiCFP Conference Series Name"]))
                    except csv.Error as e:
                        sys.exit('file {}, line {}: {}'.format(
                                self.gold_standard_file, reader.line_num, e))

            self.gold_standard = pd.DataFrame(
                    gold_std,
                    columns = ["scigraph_conferenceseries",
                               "wikicfp_conferenceseries"])
            self._save_gold_standard()

        return self.gold_standard

    def _save_gold_standard(self):
        """
        Saves the processed Gold Standard as a pickle file.
        """
        print("Saving the gold standard to disk.")
        with open(self.persistent_file, "wb") as f:
            pickle.dump(self.gold_standard, f)

    def _load_gold_standard(self):
        """
        Loads the processed Gold Standard.
        """
        if os.path.isfile(self.persistent_file):
            print("Loading gold standard.")
            with open(self.persistent_file, "rb") as f:
                self.gold_standard = pickle.load(f)
                print("Loaded.")
                return True

        return False

    def main():
        from WikiCFPLinkerGoldStandard import WikiCFPLinkerGoldStandard
        gs = WikiCFPLinkerGoldStandard()
        gs.get_gold_standard()

    if __name__ == "__main__":
        main()
