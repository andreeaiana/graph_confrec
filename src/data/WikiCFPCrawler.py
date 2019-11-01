# -*- coding: utf-8 -*-
import os
import time
import pickle
import requests
import argparse
import pandas as pd
from urllib import response
from datetime import datetime
from bs4 import BeautifulSoup, NavigableString, Tag


class WikiCFPCrawler():

    def __init__(self):
        self.base_url = "http://www.wikicfp.com/cfp/servlet/" \
                        + "event.showcfp?eventid="
        self.persistent_file_conferences = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..", "..", "data", "interim", "WikiCFP",
            "WikiCFP_conferences.pkl")
        self.persistent_file_incomplete_conferences = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..", "..", "data", "interim", "WikiCFP",
            "WikiCFP_incomplete_conferences.pkl")

    def crawl_conferences(self, start_eventid, end_eventid):
        """Crawls all WikiCFP conferences with event IDs in the given range.

        Args:
            start_eventid (int): The ID for the first conference to be crawled.
            end_eventid (int): The ID for the last conference to be crawled.

        Returns:
            dataFrame: the crawled conferences
        """
        if not self._load_conferences():
            # Crawl conferences if not crawled yet
            print("Conferences not persistent yet. Crawling now.")

            self.all_conferences = pd.DataFrame(
                    columns=['event_id', 'conference', 'name',
                             'conference_series', 'start_date', 'end_date',
                             'location', 'abstract_deadline',
                             'submission_deadline', 'notification_due',
                             'final_version_deadline', 'categories', 'link',
                             'description'])
            # Set the EventID as index
            self.all_conferences.set_index('event_id', inplace=True)

            print('Extracting the data from website.\n')

            eventid = start_eventid
            self._update_conferences(eventid, end_eventid)

            print("Finished crawling {} conferences.\n".format(len(
                    self.all_conferences)))

        else:
            # Crawl more conferences if not all have been crawled
            count_crawled_conferences = len(self.all_conferences)

            # Crawl only conferences which have not been crawled yet
            conferencesToCrawl = self._conferences_to_crawl(start_eventid,
                                                            end_eventid)
            if conferencesToCrawl:
                print("Crawling more conferences.")
                print('Extracting the data from website.\n')

                # Check which conference has been crawled last
                last_crawled = self.all_conferences.last_valid_index()

                # Check if conferences to be crawled are the immediate
                # successors of the already-crawled conferences
                if start_eventid <= last_crawled and last_crawled < end_eventid:
                    eventid = last_crawled + 1
                    self._update_conferences(eventid, end_eventid)

                else:
                    eventid = conferencesToCrawl[0]
                    end_eventid = conferencesToCrawl[-1]
                    self._update_conferences(eventid, end_eventid)

                print("Finished crawling {} conference(s).\n".format(
                        len(self.all_conferences)-count_crawled_conferences))

            else:
                # All conferences have been already crawled
                print("All conferences have been crawled already.")
                if not self._load_incomplete_conferences():
                    self.incomplete_conferences = self._get_incomplete_conferences(
                            self.all_conferences)

        print("There are {} conferences crawled, out of which {} have incomplete mandatory information.".format(
                len(self.all_conferences), len(self.incomplete_conferences)))

        return self.all_conferences

    def update_incomplete_conferences(self):
        """Updates information of conferences with incomplete mandatory
        information, if available.

        Returns:
            dataFrame: the updated conferences
        """
        if not self._load_incomplete_conferences():
            print("Incomplete conferences list not persistent yet. "
                  + "Creating now.")
            self.incomplete_conferences = self._get_incomplete_conferences(
                    self.all_conferences)

        # Update any conferences with incomplete mandatory information,
        # if possible
        if len(self.incomplete_conferences) > 0:
            updated_conferences = pd.DataFrame(
                    columns=['event_id', 'conference', 'name',
                             'conference_series', 'start_date', 'end_date',
                             'location', 'abstract_deadline',
                             'submission_deadline', 'notification_due',
                             'final_version_deadline', 'categories', 'link',
                             'description'])
            updated_conferences.set_index('event_id', inplace=True)

            for eventid in self.incomplete_conferences:
                parsed_conference = self._parse_conference(eventid)
                updated_conferences = pd.concat(
                        [updated_conferences, parsed_conference], sort=False)

            # Update the conferences dataframe with the updated conferences
            self.all_conferences.update(updated_conferences)

            # Update the list of conferences with incomplete mandatory
            # information
            self.incomplete_conferences = self._get_incomplete_conferences(
                    updated_conferences)
            self._save_incomplete_conferences()
        else:
            print('There are no conferences with incomplete mandatory '
                  + 'information.')
        return self.all_conferences

    def _update_conferences(self, eventid, end_eventid):
        """Parses more conferences for the given range.

        Args:
            eventid (int): The ID for the first conference to be crawled.
            end_eventid (int): The ID for the last conference to be crawled.

        Returns:
            dataFrame: the data frame of conferences updated with the newly
                crawled conferences
        """
        while eventid <= end_eventid:
            parsed_conference = self._parse_conference(eventid)
            if not parsed_conference.empty:
                self.all_conferences = pd.concat(
                        [self.all_conferences, parsed_conference], sort=False)
            eventid += 1

        # Store crawled conferences
        self._save_conferences()

        # Get list of event IDs for conferences with incomplete mandatory
        # information
        self.incomplete_conferences = self._get_incomplete_conferences(
                self.all_conferences)
        self._save_incomplete_conferences()

    def _parse_conference(self, eventid):
        """Parses a conference page for the given event ID.

        Args:
            eventid (int): The ID for the conference to be crawled.

        Returns:
            dataFrame: crawled conference
        """
        # Fire the request
        try:
            url = "".join([self.base_url, str(eventid)])
            print("Requesting {}".format(url))
            data = requests.get(url)
            # WikiCFP policy, issue at most one query every five seconds
            time.sleep(5)
            print("Done.")
        except Exception as e:
            print(str(e))
        if data.status_code != 200:
            print("Can't connect to WikiCFP! (status code: "
                                              + response.status_code + ")")

        conferences = pd.DataFrame(
                columns=['event_id', 'conference', 'name',
                         'conference_series', 'start_date', 'end_date',
                         'location', 'abstract_deadline',
                         'submission_deadline', 'notification_due',
                         'final_version_deadline', 'categories', 'link',
                         'description'])
        conferences.set_index('event_id', inplace=True)

        # Crawl the data
        soup = BeautifulSoup(data.text, 'lxml')

        # Get the table from the conference page
        data = self._get_table_data(soup)

        if data:
            #Extract the information from the table
            acronym = self._get_acronym(data)
            name = self._get_name(data)
            start_date = self._get_start_date(data)
            end_date = self._get_end_date(data)
            location = self._get_location(data)
            abstract_deadline = self._get_abstract_registration_deadline(data)
            submission_deadline = self._get_submission_deadline(data)
            notification_due = self._get_notification_deadline(data)
            final_version_deadline = self._get_final_version_deadline(data)
            categories = self._get_categories(data)

            # Get the remaining text from the page
            conference_series = self._get_series(soup)
            conference_link = self._get_link(soup)
            description = self._get_description(soup)

            # Add parsed data to dataframe
            conferences.loc[eventid] = [acronym, name, conference_series,
                                        start_date, end_date, location,
                                        abstract_deadline, submission_deadline,
                                        notification_due,
                                        final_version_deadline, categories,
                                        conference_link, description]
            print("Finished parsing conference.\n")
            return conferences
        else:
            print("This conference does not exist.\n")
            return conferences

    def _get_table_data(self, soup):
        """Parses table on the WikiCFP conference page.

        Args:
            soup (object): The BeatifulSoup object created for a given
                conference page.

        Returns:
            dictionary: information contained in the parsed table
        """
        table = soup.select('span[typeof="v:Event"]')
        data = {}
        if table:
            table = table[0]
            for span in table.select('span[property^="v:"]'):
                key = span["property"]
                value = span.get("content")
                if not value:
                    value = span.string
                data[key.strip()] = value.strip()

        box = soup.select('table.gglu')
        if box:
            for tr in box[0].select('tr'):
                tr_head = tr.th.string
                tr_data = tr.td
                tr_event = tr_data.select('span[typeof="v:Event"]')
                if tr_event:
                    tr_event = tr_event[0].select(
                            'span[property="v:startDate"]')
                    if tr_event:
                        tr_data = tr_event[0].get("content")
                    else:
                        tr_data = " "
                else:
                    tr_data = tr_data.string
                data[tr_head.strip()] = tr_data.strip()

            categories = list()
            for a in box[1].select("a[href]"):
                if a:
                    categories.append(a.text.strip())
            if categories:
                data[categories[0]] = categories[1:len(categories)]

        return data

    def _conferences_to_crawl(self, start_eventid, end_eventid):
        """Creates a list of conferences to be crawled based on the given range
        and the already-crawled conferences.

        Args:
            start_eventid (int): The ID for the first conference to be crawled.
            end_eventid (int): The ID for the last conference to be crawled.

        Returns:
            list: list of conference IDs that are not already crawled and are
                part of the given IDs range
        """
        conferencesToCrawl = list()
        # Check if any of the conferences is already crawled
        for eventid in range(start_eventid, end_eventid+1):
            if eventid not in self.all_conferences.index:
                conferencesToCrawl.append(eventid)
        return conferencesToCrawl

    def _get_acronym(self, data):
        acronym = data.get("v:summary").strip()
        return acronym

    def _get_name(self, data):
        text = data.get("v:description").strip()
        name = text.split(":")[1]
        return name

    def _get_start_date(self, data):
        if data.get("v:startDate"):
            date = data.get("v:startDate").strip()
            if date and date != 'TBD':
                startDate = datetime.strptime(
                        date, "%Y-%m-%dT%H:%M:%S").date().isoformat()
                return startDate
        else:
            return None

    def _get_end_date(self, data):
        if data.get("v:endDate"):
            date = data.get("v:endDate").strip()
            if date and date != 'TBD':
                endDate = datetime.strptime(
                        date, "%Y-%m-%dT%H:%M:%S").date().isoformat()
                return endDate
        else:
            return None

    def _get_location(self, data):
        if data.get("v:locality"):
            location = data.get("v:locality").strip()
            if location:
                return location
        else:
            return None

    def _get_submission_deadline(self, data):
        if data.get("Submission Deadline"):
            date = data.get("Submission Deadline").strip()
            if date and date != 'TBD':
                deadline = datetime.strptime(
                        date, "%Y-%m-%dT%H:%M:%S").date().isoformat()
                return deadline
        else:
            return None

    def _get_abstract_registration_deadline(self, data):
        if data.get("Abstract Registration Due"):
            date = data.get("Abstract Registration Due").strip()
            if date and date != 'TBD':
                deadline = datetime.strptime(
                        date, "%Y-%m-%dT%H:%M:%S").date().isoformat()
                return deadline
        else:
            return None

    def _get_notification_deadline(self, data):
        if data.get("Notification Due"):
            date = data.get("Notification Due").strip()
            if date and date != 'TBD':
                deadline = datetime.strptime(
                        date, "%Y-%m-%dT%H:%M:%S").date().isoformat()
                return deadline
        else:
            return None

    def _get_final_version_deadline(self, data):
        if data.get("Final Version Due"):
            date = data.get("Final Version Due").strip()
            if date and date != 'TBD':
                deadline = datetime.strptime(
                        date, "%Y-%m-%dT%H:%M:%S").date().isoformat()
                return deadline
        else:
            return None

    def _get_categories(self, data):
        categories = []
        if data.get('Categories'):
            categories = data.get('Categories')
        return categories

    def _get_series(self, soup):
        for content in soup.body.find_all('div', attrs = {'class': 'contsec'}):
            for table in content.find_all('tr', limit = 8):
                for infos in table.find_all('td', attrs={'align': 'center'},
                                            recursive = False):
                    if "Conference Series" in infos.text:
                        data = infos.text.strip().split('\n')
                        data = list(filter(None, data))
                        series = data[-1].split(": ")[1]
                        return series
        return None

    def _get_link(self, soup):
        for content in soup.body.find_all('div', attrs = {'class': 'contsec'}):
            for table in content.find_all('tr', limit = 8):
                for infos in table.find_all('td', attrs={'align': 'center'},
                                            recursive = False):
                    if "Link" in infos.text:
                        link = infos.a.text.strip()
                        return link
        return None

    def _get_description(self, soup):
        """Parses the description of the conference available on the WikiCFP
        conference page based on the type of HTML embedding.

        Args:
            soup (object): The BeatifulSoup object created for a given
                conference page.

        Returns:
            string: the parsed and preprocessed textual description of the
                conference
        """
        if soup.find_all('p'):
            text = self._parse_text_table(soup)
            preprocessed_text = self._preprocess_text_table(text)
        else:
            text = self._parse_text(soup)
            preprocessed_text = self._preprocess_text(text)

        return preprocessed_text

    def _parse_text(self, soup):
        """Parses the description of the conference available on the WikiCFP
        conference page if the text is not embedded in a table.

        Args:
            soup (object): The BeatifulSoup object created for a given
                conference page.

        Returns:
            string: the parsed textual description of the conference
        """
        description = ""

        for br2 in soup.find_all('br'):
            next = br2.previousSibling
            if not (next and isinstance(next, NavigableString)):
                continue
            next2 = next.previousSibling
            if next2 and isinstance(next2, Tag) and next2.name == 'br':
                text = str(next).strip()
                if text:
                    description = " ".join(
                            [description,
                             str(next.encode('ascii', 'ignore').strip())
                             ])

        if description:
            return description.split("' b\'")
        else:
            return None

    def _preprocess_text(self, description):
        """Preprocesses the textual description of a conference.

        Args:
            description (string): description of a conference

        Returns:
            string: the preprocessed textual description of the conference
        """
        text = ""

        if description:
            for line in description:
                # Replace characters and parse tabs
                line = line.replace("b'", "").replace("\\t", "\t").strip()

                # Introduce whitespaces
                if (len(line.split())<10) and ("-" not in line):
                    if not line.endswith("."):
                        line = "".join(["\n", line, "\n"])
                    else:
                        line = "".join([line, "\n"])
                text = "".join([text, "\n", line])

        if text != "":
            return text
        else:
            return None

    def _parse_text_table(self, soup):
        """Parses the description of the conference available on the WikiCFP
        conference page if the text is embedded in a table.

        Args:
            soup (object): The BeatifulSoup object created for a given
                conference page.

        Returns:
            string: the parsed textual description of the conference
        """
        description = ""
        for content in soup.body.find_all('div', attrs={'class': 'cfp'},
                                          limit=1):
            for tag in content.contents:
                if tag.name == 'p':
                    description = " ".join([
                            description, "\n",
                            str(tag.text.encode('ascii', 'ignore').strip()),
                            "\n"])
                if tag.name == 'h3':
                    description = " ".join([
                            description, "\n",
                            str(tag.text.encode('ascii', 'ignore').strip()),
                            "\n"])
                if tag.name == 'table':
                    for tr in tag.find_all('tr'):
                        description = " ".join([
                                description, "\n", str(tr.text.strip()), "\n"])
                if tag.name == 'ul':
                    for item in tag.find_all('li'):
                        description = " ".join([
                                description, "-",
                                str(item.text.encode('ascii',
                                                     'ignore').strip()), "\n"])
                        if item.a:
                            description = " ".join([
                                    description, item.a['href'],  "\n"])

        if description:
            return description.split("' b\'")
        else:
            return None

    def _preprocess_text_table(self, description):
        """Preprocesses the textual description of a conference
        (embedded as a table).

        Args:
            description (string): description of a conference

        Returns:
            string: the preprocessed textual description of the conference
        """
        text = ""
        if description:
            for line in description:
                line = line.replace("b\'", "").replace("\'", "").strip()
                text = "".join([text, "\n", line])
        return text

    def _get_incomplete_conferences(self, conferences):
        """Creates list of conferences with incomplete mandatory information.

        Args:
            conferences (dataFrame): crawled conferences

        Returns:
            list: list of conference IDs with incomplete mandatory fields
        """
        incomplete_conferences = set()
        mandatory_fields = ['start_date', 'end_date', 'location',
                            'submission_deadline']

        # Check if any of the mandatory fields is None
        for field in mandatory_fields:
            incomplete_conferences.update(
                    conferences.index[conferences[field].isnull()].tolist())
        return incomplete_conferences

    def _save_conferences(self):
        with open(self.persistent_file_conferences, "wb") as f:
            pickle.dump(self.all_conferences, f)
        csv_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "..", "..", "data", "interim", "WikiCFP",
                                "WikiCFP_conferences.csv")
        self.all_conferences.to_csv(csv_file)

    def _save_incomplete_conferences(self):
        with open(self.persistent_file_incomplete_conferences, "wb") as f:
            pickle.dump(self.incomplete_conferences, f)

    def _load_conferences(self):
        if os.path.isfile(self.persistent_file_conferences):
            print("Loading conferences...")
            with open(self.persistent_file_conferences, "rb") as f:
                self.all_conferences = pickle.load(f)
                print("Loaded.")
                return True
        return False

    def _load_incomplete_conferences(self):
        if os.path.isfile(self.persistent_file_incomplete_conferences):
            print("Loading incomplete conference list...")
            with open(self.persistent_file_incomplete_conferences,
                      "rb") as f:
                self.incomplete_conferences = pickle.load(f)
                print("Loaded.")
                return True
        return False

    def main():
        parser = argparse.ArgumentParser(
                description='Arguments for WikiCFP Crawler.')
        parser.add_argument("start_eventid",
                            type=int,
                            help="The event id from which to start crawling.")
        parser.add_argument('end_eventid',
                            type=int,
                            help="The event id at which to stop crawling.")
        args = parser.parse_args()

        from WikiCFPCrawler import WikiCFPCrawler
        print("Initializing crawler...")
        crawler = WikiCFPCrawler()
        all_conferences = crawler.crawl_conferences(args.start_eventid,
                                                    args.end_eventid)
        print("Finished.")

    if __name__ == "__main__":
        main()
