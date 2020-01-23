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
from WikiCFPCrawler import WikiCFPCrawler


class WikiCFPLinker():

    def __init__(self, similarity_metric="damerau_levenshtein",
                 match_threshold=0.885, remove_stopwords=True):
        self.crawler = WikiCFPCrawler()
        self.match_threshold = match_threshold
        self.similarity_measure = self._get_similarity_measure(
                similarity_metric)

        self.remove_stopwords = remove_stopwords
        if self.remove_stopwords:
            self.stopwords = stopwords.words("english")

        self.scigraph_series = self._load_scigraph_series()
        self.wikicfp_conf = self._load_wikicfp_conferences()
        self.wikicfp_series = self._load_wikicfp_series()

        self.matches = []
        self.scigraph_notmatched = list(
                self.scigraph_series["conferenceseries_name"])
        self.wikicfp_notmatched = list(self.wikicfp_series.values)
        self.wikicfp_names_notmatched = list(self.wikicfp_conf["name"])

        self.persistent_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..", "..", "data", "interim", "WikiCFP",
            "matched_conference_series.pkl")

        self.matched_conf_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..", "..", "data", "interim", "WikiCFP",
            "matched_conference_series.csv")

        self.scigraph_notmatched_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..", "..", "data", "interim", "WikiCFP",
            "scigraph_notmatched_series.csv")

        self.wikicfp_notmatched_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..", "..", "data", "interim", "WikiCFP",
            "wikicfp_notmatched_conf.csv")

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

    def match_conferences(self):
        """Links the conference series from SciGraph to those crawled
        from WikiCFP.
        """
        if not self._load_correspondences():
            # Link conference series that exactly match between SciGraph
            # and WikiCFP
            self._link_equally()

            # Link conference series that match between SciGraph and WikiCFP
            if len(self.scigraph_notmatched) != 0:
                self._link_series_to_series()

            for wikicfp_series in self.wikicfp_series:
                if wikicfp_series not in self.wikicfp_notmatched:
                    matched_names = self.wikicfp_conf[
                            self.wikicfp_conf[
                                    "conference_series"] == wikicfp_series
                            ]["name"].tolist()
                    for name in matched_names:
                        self.wikicfp_names_notmatched.remove(name)

            # Link SciGraph conference series to WikiCFP most recent conference
            # name belonging to a series
            if len(self.scigraph_notmatched) != 0:
                self._link_series_to_name()

            # Save matches as DataFrame and remove duplicates
            # (i.e. keep matches with highest similarity score)
            self.correspondences = pd.DataFrame(
                self.matches,
                columns=["conferenceseries", "conferenceseries_name",
                         "WikiCFP_conferenceseries", "similarity_score"]
                )
            self.correspondences = self.correspondences.sort_values(
                    "similarity_score", ascending=False).drop_duplicates(
                            subset=["conferenceseries"]).sort_index()
            self.correspondences.drop(
                    columns=["conferenceseries_name", "similarity_score"],
                    inplace=True)

            # Save correspondences
            self._save_correspondences()

            # Save remaining SciGraph conference series to be matched
            scigraph_notmatched_series = pd.DataFrame(
                    self.scigraph_notmatched,
                    columns=["conferenceseries_name"])
            scigraph_notmatched_series.to_csv(self.scigraph_notmatched_file)

            # Save remaining SciGraph conference series to be matched
            wikicfp_notmatched_series = pd.DataFrame(
                    self.wikicfp_notmatched,
                    columns=["wikicfp_conferenceseries"])
            wikicfp_notmatched_series.to_csv(self.wikicfp_notmatched_file)

        return self.correspondences

    def _link_equally(self):
        """Links the conference series from SciGraph to those from WikiCFP if
        their names are identical.
        """
        print("Computing equal matching.")
        checked = list()
        count = len(self.scigraph_series["conferenceseries_name"])

        with tqdm(desc="Linking equally: ", total=count) as pbar:
            for series_name in self.scigraph_series["conferenceseries_name"]:
                if series_name in self.scigraph_notmatched:
                    processed_scigraph_series = self._preprocess_string(
                            series_name)

                    for wikicfp_series in self.wikicfp_series:
                        if wikicfp_series in self.wikicfp_notmatched:
                            processed_wikicfp_series = self._preprocess_string(
                                    wikicfp_series)

                            if processed_scigraph_series == processed_wikicfp_series:
                                similarity = 1.0
                                sg_series = self.scigraph_series[
                                        self.scigraph_series[
                                                "conferenceseries_name"
                                                ] == series_name][
                                                "conferenceseries"].tolist()[0]
                                self.matches.append(
                                        [sg_series, series_name,
                                         wikicfp_series, similarity])
                                self.wikicfp_notmatched.remove(wikicfp_series)
                                checked.append(series_name)
                    if series_name in checked:
                        self.scigraph_notmatched.remove(series_name)
                if len(self.scigraph_notmatched) == 0:
                    break
                pbar.update(1)
        print("Equal matching computed.")

    def _link_series_to_series(self):
        """ Links the conference series from SciGraph to those from WikiCFP if
        the similarity of the conference series names, as determined by the
        chosen similarity metric, is above the chosen threshold.
        """
        print("Computing series-to-series similarities.")
        checked = list()
        count = len(self.scigraph_series["conferenceseries_name"])

        with tqdm(desc="Linking series to series: ", total=count) as pbar:
            for series_name in self.scigraph_series["conferenceseries_name"]:
                if series_name in self.scigraph_notmatched:
                    processed_scigraph_series = self._preprocess_string(
                            series_name)

                    for wikicfp_series in self.wikicfp_series:
                        if wikicfp_series in self.wikicfp_notmatched:
                            processed_wikicfp_series = self._preprocess_string(
                                    wikicfp_series)
                            similarity = self.similarity_measure(
                                    processed_scigraph_series,
                                    processed_wikicfp_series)

                            if similarity >= self.match_threshold:
                                sg_series = self.scigraph_series[
                                        self.scigraph_series[
                                                "conferenceseries_name"
                                                ] == series_name][
                                                "conferenceseries"].tolist()[0]
                                self.matches.append(
                                        [sg_series, series_name,
                                         wikicfp_series, similarity])
                                self.wikicfp_notmatched.remove(wikicfp_series)
                                checked.append(series_name)
                    if series_name in checked:
                        self.scigraph_notmatched.remove(series_name)
                if len(self.scigraph_notmatched) == 0:
                    break
                pbar.update(1)
        print("Series-to-series similarities computed.")

    def _link_series_to_name(self):
        """Links the conference series from SciGraph to the most recent
        conference from WikiCFP that belongs to a conference series, if the
        similarity of the SciGraph conference series name and of the WikiCFP
        conference name, as determined by the chosen similarity metric, is
        above the chosen threshold.
        """
        print("Computing series-to-name similarities.")
        checked = list()
        count = len(self.scigraph_series["conferenceseries_name"])

        with tqdm(desc="Linking series to name: ", total=count) as pbar:
            for series_name in self.scigraph_series["conferenceseries_name"]:
                if series_name in self.scigraph_notmatched:
                    processed_scigraph_series = self._preprocess_string(
                            series_name)

                    for wikicfp_name in self.wikicfp_conf["name"]:
                        if wikicfp_name in self.wikicfp_names_notmatched:
                            processed_wikicfp_name = self._preprocess_string(
                                    wikicfp_name)
                            similarity = self.similarity_measure(
                                    processed_scigraph_series,
                                    processed_wikicfp_name)

                            if similarity >= self.match_threshold:
                                sg_series = self.scigraph_series[
                                        self.scigraph_series[
                                                "conferenceseries_name"
                                                ] == series_name][
                                                "conferenceseries"].tolist()[0]
                                wikicfp_name = self._get_most_recent(
                                        wikicfp_name)
                                self.matches.append(
                                        [sg_series, series_name,
                                         wikicfp_name, similarity])
                            checked.append(series_name)
                    if series_name in checked:
                        self.scigraph_notmatched.remove(series_name)
                if len(self.scigraph_notmatched) == 0:
                    break
                pbar.update(1)
        print("Series-to-name similarities computed.")

    def _get_most_recent(self, wikicfp_name):
        """ Returns the most similar WikiCFP conference from several WikiCFP
        conferences with similar names.

        Args:
            conf_name (string): The name of the WikiCFP conference.

        Returns:
            string: The name of the most recent WikiCFP conference similar to
                    the given conference name.
        """
        repeating_conf = list()

        for conf_name in self.wikicfp_names_notmatched:
            similarity = self.similarity_measure(conf_name, wikicfp_name)
            if similarity >= self.match_threshold:
                conf_date = self.wikicfp_conf[
                        self.wikicfp_conf["name"] == conf_name][
                                "start_date"].tolist()[0]
                if conf_date is None:
                    conf_date = "0000-00-00"
                repeating_conf.append((conf_name, conf_date))
        most_recent = max(repeating_conf, key=operator.itemgetter(1))[0]
        for conf_name in [elem[0] for elem in repeating_conf]:
            if conf_name in self.wikicfp_names_notmatched:
                self.wikicfp_names_notmatched.remove(conf_name)

        return most_recent

    def get_statistics(self):
        """Prints statistics of the matched conference series.
        """

        print("There are {} conference series in WikiCFP.".format(
                len(self.wikicfp_series)))
        print("There are {} conference series in the SciGraph considered data.".format(
                len(self.scigraph_series)))

        percentage_matched = len(self.correspondences)/len(
                self.scigraph_series)
        print("{} out of {} conference series have been matched, i.e. {}".format(
                len(self.correspondences), len(self.scigraph_series), percentage_matched))

    def _preprocess_string(self, string):
        """Removes stopwords from a given string.

        Args:
            string (string): The string from which to remove the stopwords.

        Returns:
            string: The string without stopwords.
        """
        string = string.lower()
        if self.remove_stopwords:
            string = ' '.join([word for word in string.split() if word
                               not in self.stopwords])
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

    def _load_scigraph_series(self):
        """
        Loads the training and test data for the given data size.

        Args:
            data_name (string): The size of the data to be loaded.

        Returns:
            dataFrame: The concatenated conference series for the training and
            test data.
        """
        d_train = DataLoader()
        d_train.training_data_with_abstracts_citations()
        training_conference_series = d_train.data[["conferenceseries",
                                                   "conferenceseries_name"]]
        d_validation = DataLoader()
        d_validation.validation_data_with_abstracts_citations()
        validation_conference_series = d_validation.data[[
                "conferenceseries", "conferenceseries_name"]]
        d_test = DataLoader()
        d_test.test_data_with_abstracts_citations()
        test_conference_series = d_test.data[["conferenceseries",
                                              "conferenceseries_name"]]

        scigraph_series = pd.concat([training_conference_series,
                                     validation_conference_series,
                                     test_conference_series])
        scigraph_series = scigraph_series.drop_duplicates()

        return scigraph_series

    def _load_wikicfp_conferences(self):
        """Loads the WikiCFP conferences names and start dates.

        Returns:
            dataFrame: The WikiCFP conferences names and start dates.
        """
        self.crawler._load_conferences()
        wikicfp_conf = self.crawler.all_conferences[[
                "name", "conference_series", "start_date"]].drop_duplicates()
        return wikicfp_conf

    def _load_wikicfp_series(self):
        """Loads the WikiCFP conference series.

        Returns:
            dataFrame: The WikiCFP conference series.
        """
        self.crawler._load_conferences()
        wikicfp_series = self.crawler.all_conferences[
                ~pd.isnull(self.crawler.all_conferences["conference_series"])
                ]["conference_series"].drop_duplicates()
        return wikicfp_series

    def _load_correspondences(self):
        """Loads the computed correspondences between the SciGraph and the
        WikiCFP conference series.
        """
        if os.path.isfile(self.persistent_file):
            print("Loading correspondences...")
            with open(self.persistent_file, "rb") as f:
                self.correspondences = pickle.load(f)
                print("Loaded.")
                return True

        return False

    def _save_correspondences(self):
        """Saves the computed correspondences between the SciGraph and the
        WikiCFP conference series.
        """
        self.correspondences.to_csv(self.matched_conf_file)
        print("Saving to disk.")
        with open(self.persistent_file, "wb") as f:
            pickle.dump(self.correspondences, f)

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
                            help='Whether to remove stopwords.')
        args = parser.parse_args()
        print("Starting...")
        from WikiCFPLinker import WikiCFPLinker
        linker = WikiCFPLinker(args.similarity_metric, args.match_threshold,
                               args.remove_stopwords)
        linker.match_conferences()
        print("Finished.")

    if __name__ == "__main__":
        main()
