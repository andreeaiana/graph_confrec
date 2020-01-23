# -*- coding: utf-8 -*-
import os
import pickle
import operator
import argparse
import pandas as pd
from tqdm import tqdm
import jellyfish as jf
from nltk.corpus import stopwords
from DataLoader import DataLoader
from H5IndexScraper import Scraper


class H5IndexLinker:

    def __init__(self, similarity_metric="damerau_levenshtein",
                 threshold=0.9):
        self.threshold = threshold
        self.similarity_measure = self._get_similarity_measure(
                similarity_metric)
        self.stopwords = stopwords.words("english")

        self.conferenceseries = self._load_conferenceseries()
        self.rankings = self._load_rankings()

        self.matches = list()
        self.notmatched = list(self.conferenceseries.conferenceseries_name)

        self.persistent_file_pkl = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..", "..", "data", "interim", "H5Index",
            "matched_conferenceseries.pkl")

        self.persistent_file_csv = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..", "..", "data", "interim", "H5Index",
            "matched_conferenceseries.csv")

        self.notmatched_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..", "..", "data", "interim", "H5Index",
            "notmatched_conferenceseries.csv")

    def match_conferences(self):
        if not self._load_correspondences():
            print("Correspondences not computed yet. Computing now...")
            self._link_equally()
            if not len(self.notmatched) == 0:
                self._link_similar()
            print("Correspondences computed.")

            # Save matches as DataFrame and remove duplicates
            # (i.e. keep matches with highest similarity score)
            self.correspondences = pd.DataFrame(
                self.matches,
                columns=["conferenceseries", "conferenceseries_name",
                         "publication", "category", "subcategory", "h5_index",
                         "h5_median", "similarity"])
            self.correspondences = self.correspondences.sort_values(
                    "similarity", ascending=False)
            self.correspondences = self.correspondences.drop_duplicates(
                    subset=["conferenceseries"]).reset_index(drop=True)
            self.correspondences.drop(columns=["similarity"],
                                      inplace=True)

            # Save correspondences
            self._save_correspondences()

            # Save remaining SciGraph conference series to be matched
            notmatched_series = pd.DataFrame(self.notmatched,
                                             columns=["conferenceseries_name"])
            notmatched_series.to_csv(self.notmatched_file)
        else:
            print("Correspondences already computed.")
        self.get_statistics()
        return self.correspondences

    def _link_equally(self):
        print("Computing equal matching...")
        checked = list()
        count_conf = len(self.conferenceseries)
        with tqdm(desc="Linking equally: ", total=count_conf) as pbar:
            for conf_name in self.conferenceseries.conferenceseries_name:
                if conf_name in self.notmatched:
                    processed_conf_name = self._preprocess_string(conf_name)
                    for pub_name in self.rankings.publication:
                        processed_pub_name = self._preprocess_string(pub_name)
                        if processed_conf_name == processed_pub_name:
                            conf_series = self.conferenceseries[
                                    self.conferenceseries.conferenceseries_name
                                    == conf_name].conferenceseries.tolist()[0]
                            idx = self.rankings[
                                    self.rankings.publication == pub_name
                                    ].index.tolist()[0]
                            self.matches.append([
                                    conf_series,
                                    conf_name,
                                    pub_name,
                                    self.rankings.loc[idx].category,
                                    self.rankings.loc[idx].subcategory,
                                    self.rankings.loc[idx].h5_index,
                                    self.rankings.loc[idx].h5_median,
                                    1.0])
                            checked.append(conf_name)
                    if conf_name in checked:
                        self.notmatched.remove(conf_name)
                if len(self.notmatched) == 0:
                    break
                pbar.update(1)
        print("Computed.")

    def _link_similar(self):
        print("Computing matching based on similarity...")
        checked = list()
        count_conf = len(self.conferenceseries)
        with tqdm(desc="Linking similar: ", total=count_conf) as pbar:
            for conf_name in self.conferenceseries.conferenceseries_name:
                if conf_name in self.notmatched:
                    processed_conf_name = self._preprocess_string(conf_name)
                    for pub_name in self.rankings.publication:
                        processed_pub_name = self._preprocess_string(pub_name)
                        similarity = self.similarity_measure(
                                processed_conf_name, processed_pub_name)
                        if similarity >= self.threshold:
                            conf_series = self.conferenceseries[
                                    self.conferenceseries.conferenceseries_name
                                    == conf_name].conferenceseries.tolist()[0]
                            idx = self.rankings[
                                    self.rankings.publication == pub_name
                                    ].index.tolist()[0]
                            self.matches.append([
                                    conf_series,
                                    conf_name,
                                    pub_name,
                                    self.rankings.loc[idx].category,
                                    self.rankings.loc[idx].subcategory,
                                    self.rankings.loc[idx].h5_index,
                                    self.rankings.loc[idx].h5_median,
                                    similarity])
                            checked.append(conf_name)
                    if conf_name in checked:
                        self.notmatched.remove(conf_name)
                if len(self.notmatched) == 0:
                    break
                pbar.update(1)
        print("Computed.")

    def _get_similarity_measure(self, similarity_metric):
        """Returns the similarity measure for the chosen metric.

        Args:
            similarity_metric (string): The similarity metric to be used.

        Returns:
            method: The chosen similarity measure.
        """
        metric_name = "_" + str(similarity_metric) + "_match"
        similarity_measure = getattr(self, metric_name)
        return similarity_measure

    def _preprocess_string(self, string):
        """Removes stopwords from a given string.

        Args:
            string (string): The string from which to remove the stopwords.

        Returns:
            string: The string without stopwords.
        """
        string = string.lower()
        string = ' '.join([word for word in string.split() if word not in
                           self.stopwords])
        return string

    def _levenshtein_match(self, string1, string2):
        """Computes the Levenshtein similarity between two strings.

        Args:
            string1 (string): First string to be considered.
            string2 (string): Second string to be considered.

        Returns:
            boolean: True, if the similarity is above a given thereshold,
            false otherwise.
        """
        distance = jf.levenshtein_distance(string1, string2)
        if len(string1) == 0 and len(string2) == 0:
            similarity = 0
        else:
            similarity = 1-distance/max(len(string1), len(string2))
        return similarity

    def _damerau_levenshtein_match(self, string1, string2):
        """Computes the Damerau Levenshtein similarity between two strings.

        Args:
            string1 (string): First string to be considered.
            string2 (string): Second string to be considered.

        Returns:
            boolean: True, if the similarity is above a given thereshold,
            false otherwise.
        """
        distance = jf.damerau_levenshtein_distance(string1, string2)
        if len(string1) == 0 and len(string2) == 0:
            similarity = 0
        else:
            similarity = 1-distance/max(len(string1), len(string2))
        return similarity

    def _jaro_match(self, string1, string2):
        """Computes the Jaro similarity between two strings.

        Args:
            string1 (string): First string to be considered.
            string2 (string): Second string to be considered.

        Returns:
            boolean: True, if the similarity is above a given thereshold,
            false otherwise.
        """
        similarity = jf.jaro_distance(string1, string2)
        return similarity

    def _jaro_winkler_match(self, string1, string2):
        """Computes the Jaro-Winkler similarity between two strings.

        Args:
            string1 (string): First string to be considered.
            string2 (string): Second string to be considered.

        Returns:
            boolean: True, if the similarity is above a given thereshold,
            false otherwise.
        """
        similarity = jf.jaro_winkler(string1, string2)
        return similarity

    def get_statistics(self):
        print("There are {} conference series in SciGraph.".format(
                len(self.conferenceseries)))
        print("There are {} publications ranked by Google H5Index.".format(
                len(self.rankings)))
        percentage_matched = len(self.correspondences)/len(
                self.conferenceseries)
        print("{}/{} conference series have been matched (i.e. {}).".format(
                len(self.correspondences), len(self.conferenceseries),
                percentage_matched))

    def _load_conferenceseries(self):
        """
        Loads the training, validation and test data conference series.

        Returns:
            dataFrame: The concatenated conference series for the training,
            validation and test data.
        """
        print("Loading conference series...")
        d_train = DataLoader()
        df_train = d_train.training_data_with_abstracts_citations().data

        d_val = DataLoader()
        df_validation = d_val.validation_data_with_abstracts_citations().data

        d_test = DataLoader()
        df_test = d_test.validation_data_with_abstracts_citations().data

        data = pd.concat((df_train, df_validation, df_test),
                         axis=0).reset_index(drop=True)
        conferenceseries = data[["conferenceseries", "conferenceseries_name"]]
        conferenceseries.drop_duplicates(["conferenceseries"], inplace=True)
        print("Loaded {} conference series.".format(len(conferenceseries)))

        return conferenceseries

    def _load_rankings(self):
        """Loads the H5Index rankings.

        Returns:
            dataFrame: The rankings.
        """
        print("Loading rankings.")
        scraper = Scraper()
        scraper._load_rankings()
        rankings = scraper.rankings
        print("Loaded {} rankings.".format(len(rankings)))
        return rankings

    def _load_correspondences(self):
        """Loads the computed correspondences between the conference series
        and their H5Index rankings.
        """
        if os.path.isfile(self.persistent_file_pkl):
            print("Loading correspondences...")
            with open(self.persistent_file_pkl, "rb") as f:
                self.correspondences = pickle.load(f)
                print("Loaded.")
                return True
        return False

    def _save_correspondences(self):
        """Saves the computed correspondences between the conference series
        and their H5Index rankings.
        """
        self.correspondences.to_csv(self.persistent_file_csv)
        print("Saving to disk...")
        with open(self.persistent_file_pkl, "wb") as f:
            pickle.dump(self.correspondences, f)
        print("Saved.")

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for H5IndexLinker.')
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
        from H5IndexLinker import H5IndexLinker
        linker = H5IndexLinker(args.similarity_metric, args.threshold)
        linker.match_conferences()
        print("Finished.")

    if __name__ == "__main__":
        main()
