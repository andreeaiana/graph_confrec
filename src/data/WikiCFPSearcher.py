# -*- coding: utf-8 -*-
import os
import pickle
from collections import defaultdict
from WikiCFPCrawler import WikiCFPCrawler
from WikiCFPCrawler import WikiCFPCrawler
from WikiCFPLinker import WikiCFPLinker


class WikiCFPSearcher:

    def __init__(self, threshold_date="2020-01-01"):
        self.threshold_date = threshold_date

        crawler = WikiCFPCrawler()
        crawler._load_conferences()
        self.wikicfp_conf = crawler.all_conferences
        self.wikicfp_series = self.wikicfp_conf.conference_series.drop_duplicates()

        linker = WikiCFPLinker()
        linker._load_correspondences()
        self.correspondeces = linker.correspondences

        self.wikicfp_data = defaultdict()
        self.persistent_file = os.path.join("..", "..", "data", "interim",
                                            "WikiCFP", "wikicfp_data.pkl")

    def retrieve_info(self):
        if not self._load_wikicfp_data():
            print("Searching through WikiCFP...")
            self._search_correspondences()
            self._save_wikicfp_data()
        print("Number of conferences with submission deadline after {}: {}.".format(
                self.threshold_date, len(self.wikicfp_data.keys())))

    def _search_correspondences(self):
        matched_conferences = self.correspondeces.conferenceseries
        for conf in matched_conferences:
            wikicfp_conf = self.correspondeces[
                    self.correspondeces.conferenceseries == conf][
                            "WikiCFP_conferenceseries"].tolist()[0]
            if wikicfp_conf in list(self.wikicfp_series):
                conf_id = self._get_latest_conference(wikicfp_conf)
            else:
                conf_id = self.wikicfp_conf[
                        self.wikicfp_conf.name == wikicfp_conf.encode(
                                "utf-8").decode("unicode_escape")
                        ].index.tolist()
                if conf_id:
                    conf_id = conf_id[0]
            if conf_id and conf_id is not None:
                info = self._get_info(conf_id)
                if self._check_period(info["submission_deadline"]):
                    self.wikicfp_data[conf] = info

    def _get_info(self, conf_id):
        acronym = self.wikicfp_conf.loc[conf_id].conference
        name = self.wikicfp_conf.loc[conf_id]["name"]
        conf_series = self.wikicfp_conf.loc[conf_id].conference_series
        start_date = self.wikicfp_conf.loc[conf_id].start_date
        end_date = self.wikicfp_conf.loc[conf_id].end_date
        location = self.wikicfp_conf.loc[conf_id].location
        abstract_deadline = self.wikicfp_conf.loc[conf_id].abstract_deadline
        submission_deadline = self.wikicfp_conf.loc[
                conf_id].submission_deadline
        notification_due = self.wikicfp_conf.loc[conf_id].notification_due
        final_version_deadline = self.wikicfp_conf.loc[
                conf_id].final_version_deadline
        categories = self.wikicfp_conf.loc[conf_id].categories
        description = self.wikicfp_conf.loc[conf_id].description
        external_link = self.wikicfp_conf.loc[conf_id].link
        wikicfp_link = "http://www.wikicfp.com/cfp/servlet/event.showcfp?eventid="\
                       + str(conf_id)

        info = {
                "acronym": acronym,
                "name": name,
                "conference_series": conf_series,
                "start_date": start_date,
                "end_date": end_date,
                "location": location,
                "abstract_deadline": abstract_deadline,
                "submission_deadline": submission_deadline,
                "notification_due": notification_due,
                "final_version_due": final_version_deadline,
                "categories": categories,
                "description": description,
                "external_link": external_link,
                "wikicfp_link": wikicfp_link
                }
        return info

    def _get_latest_conference(self, conf_series):
        dates = self.wikicfp_conf[
                self.wikicfp_conf.conference_series == conf_series
                ].start_date.tolist()
        dates = [date for date in dates if date is not None]
        if dates:
            most_recent_date = max(dates)
            index = self.wikicfp_conf[
                    (self.wikicfp_conf.conference_series == conf_series) &
                    (self.wikicfp_conf.start_date == most_recent_date)
                    ].index.tolist()[0]
        else:
            index = None
        return index

    def _check_period(self, start_date):
        if start_date is None:
            return False
        else:
            return start_date >= self.threshold_date

    def _save_wikicfp_data(self):
        print("Saving to disk...")
        with open(self.persistent_file, "wb") as f:
            pickle.dump(self.wikicfp_data, f)
        print("Saved.")

    def _load_wikicfp_data(self):
        if os.path.isfile(self.persistent_file):
            print("Loading WikiCFP data...")
            with open(self.persistent_file, "rb") as f:
                self.wikicfp_data = pickle.load(f)
                print("Loaded.")
                return True
        return False

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for WikiCFPSearcher.')
        parser.add_argument('--threshold_date',
                            default="2020-01-01",
                            help="Threshold date for retrieving info.")
        args = parser.parse_args()
        print("Starting...")
        from WikiCFPSearcher import WikiCFPSearcher
        searcher = WikiCFPSearcher(args.threshold_date)
        searcher.retrieve_info()
        print("Finished.")

    if __name__ == "__main__":
        main()
