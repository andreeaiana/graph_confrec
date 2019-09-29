# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:57:31 2019

@author: Andreea
"""
import io
import os
import gzip
import json
import pickle
import tarfile
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.realpath(__file__), "..", "..",
                                "utils"))
from TimerCounter import Timer

# Dataset names
books_file = "books.tar.gz"
chapters_file = "chapters.tar.gz"
authors_file = "persons.tar.gz"
old_books_file = "springernature-scigraph-books.cc-by.2017-11-07.nt.gz"
old_conferences_file = "springernature-scigraph-conferences.cc-zero." \
                        + "2017-11-07-UPDATED.nt.gz"

# Book attributes (old dataset)
nt_book = "<http://scigraph.springernature.com/things/books/"
nt_has_conference = "<http://scigraph.springernature.com/ontologies/core/" \
                    + "hasConference>"
nt_shortTitle = "<http://scigraph.springernature.com/ontologies/core/" \
                + "shortTitle>"
nt_webpage = "<http://scigraph.springernature.com/ontologies/core/webpage>"

# Conference attributes (old dataset)
nt_conferences = "<http://scigraph.springernature.com/things/conferences/"
nt_acronym = "<http://scigraph.springernature.com/ontologies/core/acronym>"
nt_city = "<http://scigraph.springernature.com/ontologies/core/city>"
nt_country = "<http://scigraph.springernature.com/ontologies/core/country>"
nt_dateend = "<http://scigraph.springernature.com/ontologies/core/dateEnd>"
nt_datestart = "<http://scigraph.springernature.com/ontologies/core/dateStart>"
nt_year = "<http://scigraph.springernature.com/ontologies/core/year>"
nt_has_conference_series = "<http://scigraph.springernature.com/ontologies/" \
                            + "core/hasConferenceSeries>"

# Conference series attributes (old dataset)
nt_conference_series = "<http://scigraph.springernature.com/things/" \
                        + "conference-series/"
nt_name = "<http://scigraph.springernature.com/ontologies/core/name>"

# Parse data only for these years
years = ["2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015",
         "2016", "2017", "2018"]


class FileParser:

    path_raw = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "..", "..", "data", "raw")
    path_persistent = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "..", "..", "data", "interim", "parsed_data"
                                   )

    def __init__(self):
        self.timer = Timer()
        self.persistent = {}
        self.processes = {
                # Old datasets
                "old_books": {
                        "filename": os.path.join(self.path_raw,
                                                 old_books_file),
                        "process_line": "_process_line_old_books",
                        "persistent_file": os.path.join(self.path_persistent,
                                                        "old_books.pkl"),
                        "persistent_variable": [],
                        "dataset_format": "ntriples"
                        },
                "old_books_new_books": {
                        "filename": os.path.join(self.path_raw,
                                                 old_books_file),
                        "process_line": "_process_line_old_books_new_books",
                        "persistent_file": os.path.join(
                                self.path_persistent,
                                "old_books_new_books.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "ntriples"
                        },
                "books_conferences": {
                        "filename": os.path.join(self.path_raw,
                                                 old_books_file),
                        "process_line": "_process_line_books_conferences",
                        "persistent_file": os.path.join(
                                self.path_persistent,
                                "books_conferences.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "ntriples"
                        },
                "conferences": {
                        "filename": os.path.join(self.path_raw,
                                                 old_conferences_file),
                        "process_line": "_process_line_conferences",
                        "persistent_file": os.path.join(self.path_persistent,
                                                        "conferences.pkl"),
                        "persistent_variable": [],
                        "dataset_format": "ntriples"
                        },
                "conferences_name": {
                        "filename": os.path.join(self.path_raw,
                                                 old_conferences_file),
                        "process_line": "_process_line_conferences_name",
                        "persistent_file": os.path.join(
                                self.path_persistent, "conferences_name.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "ntriples"
                        },
                "conferences_acronym": {
                        "filename": os.path.join(self.path_raw,
                                                 old_conferences_file),
                        "process_line": "_process_line_conferences_acronym",
                        "persistent_file": os.path.join(
                                self.path_persistent,
                                "conferences_acronym.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "ntriples"
                        },
                "conferences_city": {
                        "filename": os.path.join(self.path_raw,
                                                 old_conferences_file),
                        "process_line": "_process_line_conferences_city",
                        "persistent_file": os.path.join(
                                self.path_persistent,
                                "conferences_city.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "ntriples"
                        },
                "conferences_country": {
                        "filename": os.path.join(self.path_raw,
                                                 old_conferences_file),
                        "process_line": "_process_line_conferences_country",
                        "persistent_file": os.path.join(
                                self.path_persistent,
                                "conferences_country.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "ntriples"
                        },
                "conferences_year": {
                        "filename": os.path.join(self.path_raw,
                                                 old_conferences_file),
                        "process_line": "_process_line_conferences_year",
                        "persistent_file": os.path.join(
                                self.path_persistent,
                                "conferences_year.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "ntriples"
                        },
                "conferences_datestart": {
                        "filename": os.path.join(self.path_raw,
                                                 old_conferences_file),
                        "process_line": "_process_line_conferences_datestart",
                        "persistent_file": os.path.join(
                                self.path_persistent,
                                "conferences_datestart.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "ntriples"
                        },
                "conferences_dateend": {
                        "filename": os.path.join(self.path_raw,
                                                 old_conferences_file),
                        "process_line": "_process_line_conferences_dateend",
                        "persistent_file": os.path.join(
                                self.path_persistent,
                                "conferences_dateend.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "ntriples"
                        },
                "conferences_conferenceseries": {
                        "filename": os.path.join(self.path_raw,
                                                 old_conferences_file),
                        "process_line": "_process_line_conferences_conferenceseries",
                        "persistent_file": os.path.join(
                                self.path_persistent,
                                "conferences_conferenceseries.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "ntriples"
                        },
                "conferenceseries": {
                        "filename": os.path.join(self.path_raw,
                                                 old_conferences_file),
                        "process_line": "_process_line_conferenceseries",
                        "persistent_file": os.path.join(
                                self.path_persistent, "conferenceseries.pkl"),
                        "persistent_variable": [],
                        "dataset_format": "ntriples"
                        },
                "conferenceseries_name": {
                        "filename": os.path.join(self.path_raw,
                                                 old_conferences_file),
                        "process_line": "_process_line_conferenceseries_name",
                        "persistent_file": os.path.join(
                                self.path_persistent,
                                "conferenceseries_name.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "ntriples"
                        },

                # New datasets
                "books": {
                        "filename": os.path.join(self.path_raw, books_file),
                        "process_line": "_process_line_books",
                        "persistent_file": os.path.join(self.path_persistent,
                                                        "books.pkl"),
                        "persistent_variable": [],
                        "dataset_format": "json"
                        },
                "books_title": {
                        "filename": os.path.join(self.path_raw, books_file),
                        "process_line": "_process_line_books_title",
                        "persistent_file": os.path.join(self.path_persistent,
                                                        "books_title.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "json"
                        },
                "books_year": {
                        "filename": os.path.join(self.path_raw, books_file),
                        "process_line": "_process_line_books_year",
                        "persistent_file": os.path.join(self.path_persistent,
                                                        "books_year.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "json"
                        },
                "books_language": {
                        "filename": os.path.join(self.path_raw, books_file),
                        "process_line": "_process_line_books_language",
                        "persistent_file": os.path.join(self.path_persistent,
                                                        "books_language.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "json"
                        },
                "books_abstract": {
                        "filename": os.path.join(self.path_raw, books_file),
                        "process_line": "_process_line_books_abstract",
                        "persistent_file": os.path.join(self.path_persistent,
                                                        "books_abstract.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "json"
                        },
                "books_keywords": {
                        "filename": os.path.join(self.path_raw, books_file),
                        "process_line": "_process_line_books_keywords",
                        "persistent_file": os.path.join(self.path_persistent,
                                                        "books_keywords.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "json"
                        },
                "books_authors": {
                        "filename": os.path.join(self.path_raw, books_file),
                        "process_line": "_process_line_books_authors",
                        "persistent_file": os.path.join(self.path_persistent,
                                                        "books_authors.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "json"
                        },
                "authors": {
                        "filename": os.path.join(self.path_raw, authors_file),
                        "process_line": "_process_line_authors",
                        "persistent_file": os.path.join(self.path_persistent,
                                                        "authors.pkl"),
                        "persistent_variable": [],
                        "dataset_format": "json"
                        },
                "authors_name": {
                        "filename": os.path.join(self.path_raw, authors_file),
                        "process_line": "_process_line_authors_name",
                        "persistent_file": os.path.join(self.path_persistent,
                                                        "authors_name.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "json"
                        },
                "chapters": {
                        "filename": os.path.join(self.path_raw, chapters_file),
                        "process_line": "_process_line_chapters",
                        "persistent_file": os.path.join(self.path_persistent,
                                                        "chapters.pkl"),
                        "persistent_variable": [],
                        "dataset_format": "json"
                        },
                "chapters_title": {
                        "filename": os.path.join(self.path_raw, chapters_file),
                        "process_line": "_process_line_chapters_title",
                        "persistent_file": os.path.join(self.path_persistent,
                                                        "chapters_title.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "json"
                        },
                "chapters_year": {
                        "filename": os.path.join(self.path_raw, chapters_file),
                        "process_line": "_process_line_chapters_year",
                        "persistent_file": os.path.join(self.path_persistent,
                                                        "chapters_year.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "json"
                        },
                "chapters_language": {
                        "filename": os.path.join(self.path_raw, chapters_file),
                        "process_line": "_process_line_chapters_language",
                        "persistent_file": os.path.join(
                                self.path_persistent, "chapters_language.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "json"
                        },
                "chapters_abstract": {
                        "filename": os.path.join(self.path_raw, chapters_file),
                        "process_line": "_process_line_chapters_abstract",
                        "persistent_file": os.path.join(
                                self.path_persistent, "chapters_abstract.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "json"
                        },
                "chapters_authors": {
                        "filename": os.path.join(self.path_raw, chapters_file),
                        "process_line": "_process_line_chapters_authors",
                        "persistent_file": os.path.join(
                                self.path_persistent, "chapters_authors.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "json"
                        },
                "chapters_citations": {
                        "filename": os.path.join(self.path_raw, chapters_file),
                        "process_line": "_process_line_chapters_citations",
                        "persistent_file": os.path.join(
                                self.path_persistent,
                                "chapters_citations.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "json"
                        },
                "chapters_keywords": {
                        "filename": os.path.join(self.path_raw, chapters_file),
                        "process_line": "_process_line_chapters_keywords",
                        "persistent_file": os.path.join(
                                self.path_persistent, "chapters_keywords.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "json"
                        },
                "chapters_books": {
                        "filename": os.path.join(self.path_raw, chapters_file),
                        "process_line": "_process_line_chapters_books",
                        "persistent_file": os.path.join(self.path_persistent,
                                                        "chapters_books.pkl"),
                        "persistent_variable": {},
                        "dataset_format": "json"
                        },
                }

    def get_data(self, process):
        # Check if the data is already present
        if (process in self.persistent):
            return self.persistent[process]

        print("Process '{}' not in memory yet.".format(process))

        # Load from persistent file if data already processed
        if os.path.isfile(self.processes[process]["persistent_file"]):
            with open(self.processes[process]["persistent_file"],
                      "rb") as f:
                self.persistent[process] = pickle.load(f)
                return self.persistent[process]

        print("Process '{}' not persistent yet. Processing.".format(
                process))

        # Process the raw data
        self.persistent[process] = self.processes[process][
                "persistent_variable"]
        self._parse_file(
                self.processes[process]["filename"],
                self.processes[process]["process_line"],
                self.persistent[process],
                self.processes[process]["dataset_format"]
                )
        with open(self.processes[process]["persistent_file"], "wb") as f:
            pickle.dump(self.persistent[process], f)

        return self.persistent[process]

    def _parse_file(self, filename, process_line, results, dataset_format):
        if dataset_format == "json":
            self._process_json_file(filename, process_line, results)
        else:
            self._process_ntriples_file(filename, process_line, results)

    def _process_json_file(self, filename, process_line, results):
        print("Computing number of json files.")
        with tarfile.open(filename, "r:gz", encoding="utf-8") as tar:
            count_files = len(tar.getnames())
        print("Finished computing number of files: {}.\n".format(
                count_files))

        print("Start processing file.\n")
        self.timer.tic()
        process_line_function = self.__getattribute__(process_line)
        with tqdm(desc="Processing files: ", total=count_files,
                  unit="file") as pbar:
            with tarfile.open(filename, "r:gz", encoding="utf-8") as tar:
                for member in tar.getmembers():
                    if "jsonl" in member.name:
                        file = tar.extractfile(member)
                        content = [json.loads(line) for line in
                                   file.readlines()]
                        for line in content:
                            process_line_function(line, results)
                    pbar.update(1)
        self.timer.toc()
        print("Finished processing file.\n\n")

    def _process_ntriples_file(self, filename, process_line, results):
        print("Computing file size.")
        with gzip.open(filename, mode="rt", encoding="utf-8") as f:
            file_size = f.seek(0, io.SEEK_END)
        print("Finished computing file size: {} bytes.\n".format(
                file_size))

        print("Start processing file.\n")
        self.timer.tic()
        process_line_function = self.__getattribute__(process_line)
        with tqdm(desc="Processing file: ", total=file_size,
                  unit="bytes") as pbar:
            with gzip.open(filename, mode="rt", encoding="utf-8") as f:
                for line in f:
                    process_line_function(line, results)
                    pbar.update(len(line))
        self.timer.toc()
        print("Finished processing file.\n\n")

# Processes implementations
    def _process_line_old_books(self, line, results):
        line = line.rstrip(" .\n").split(maxsplit=2)
        if line[1] == nt_has_conference:
            if line[0].startswith(nt_book):
                if line[0] not in results:
                    results.append(line[0])

    def _process_line_old_books_new_books(self, line, results):
        line = line.rstrip(" .\n").split(maxsplit=2)
        if line[1] == nt_webpage:
            if line[0].startswith(nt_book):
                new_book_id = "sg:pub." + line[2].split(".com/")[-1].rsplit(
                        ">")[0]
                results[line[0]] == new_book_id

    def _process_line_books_conferences(self, line, results):
        line = line.rstrip(" .\n").split(maxsplit=2)
        if line[1] == nt_has_conference:
            if line[0].startswith(nt_book):
                results[line[0]] = line[2]

    def _process_line_conferences(self, line, results):
        line = line.rstrip(" .\n").split(maxsplit=2)
        if line[0].startswith(nt_conferences):
            if line[0] not in results:
                results.append(line[0])

    def _process_line_conferences_name(self, line, results):
        line = line.rstrip(" .\n").split(maxsplit=2)
        if line[1] == nt_name:
            if line[0].startswith(nt_conferences):
                results[line[0]] = line[2]

    def _process_line_conferences_acronym(self, line, results):
        line = line.rstrip(" .\n").split(maxsplit=2)
        if line[1] == nt_acronym:
            if line[0].startswith(nt_conferences):
                results[line[0]] = line[2]

    def _process_line_conferences_city(self, line, results):
        line = line.rstrip(" .\n").split(maxsplit=2)
        if line[1] == nt_city:
            if line[0].startswith(nt_conferences):
                results[line[0]] = line[2]

    def _process_line_conferences_country(self, line, results):
        line = line.rstrip(" .\n").split(maxsplit=2)
        if line[1] == nt_country:
            if line[0].startswith(nt_conferences):
                results[line[0]] = line[2]

    def _process_line_conferences_year(self, line, results):
        line = line.rstrip(" .\n").split(maxsplit=2)
        if line[1] == nt_year:
            if line[0].startswith(nt_conferences):
                results[line[0]] = line[2]

    def _process_line_conferences_datestart(self, line, results):
        line = line.rstrip(" .\n").split(maxsplit=2)
        if line[1] == nt_datestart:
            if line[0].startswith(nt_conferences):
                results[line[0]] = line[2]

    def _process_line_conferences_dateend(self, line, results):
        line = line.rstrip(" .\n").split(maxsplit=2)
        if line[1] == nt_dateend:
            if line[0].startswith(nt_conferences):
                results[line[0]] = line[2]

    def _process_line_conferences_conferenceseries(self, line, results):
        line = line.rstrip(" .\n").split(maxsplit=2)
        if line[1] == nt_has_conference_series:
            results[line[0]] = line[2]

    def _process_line_conferenceseries(self, line, results):
        line = line.rstrip(" .\n").split(maxsplit=2)
        if line[0].startswith(nt_conference_series):
            if line[0] not in results:
                results.append(line[0])

    def _process_line_conferenceseries_name(self, line, results):
        line = line.rstrip(" .\n").split(maxsplit=2)
        if line[1] == nt_name:
            if line[0].startswith(nt_conference_series):
                results[line[0]] = line[2]

    def _process_line_books(self, line, results):
        if line["id"] not in results:
            results.append(line["id"])

    def _process_line_books_title(self, line, results):
        if "name" in line.keys():
            results[line["id"]] = line["name"]

    def _process_line_books_year(self, line, results):
        if "datePublished" in line.keys():
            year = line["datePublished"].split("-")[0]
            results[line["id"]] = year

    def _process_line_books_language(self, line, results):
        if "inLanguage" in line.keys():
            results[line["id"]] = line["inLanguage"]

    def _process_line_books_abstract(self, line, results):
        if "description" in line.keys():
            results[line["id"]] = line["description"]

    def _process_line_books_keywords(self, line, results):
        if "keywords" in line.keys():
            results[line["id"]] = line["keywords"]

    def _process_line_books_authors(self, line, results):
        if "author" in line.keys():
            authors = line["author"]
            author_names = list()
            for i in range(len(authors)):
                family_name = authors[i]["familyName"] if "familyName" \
                            in authors[i].keys() else ""
                given_name = authors[i]["givenName"] if "givenName" \
                            in authors[i].keys() else ""
                author_names.append(family_name + " " + given_name)
            results[line["id"]] = author_names

    def _process_line_authors(self, line, results):
        if line["id"] not in results:
            results.append(line["id"])

    def _process_line_authors_name(self, line, results):
        family_name = line["familyName"] if "familyName" in line.keys() else ""
        given_name = line["givenName"] if "givenName" in line.keys() else ""
        if not family_name == "Not available":
            author_name = family_name + " " + given_name
        else:
            author_name = ""
        results[line["id"]] = author_name

    def _process_line_chapters(self, line, results):
        if line["id"] not in results:
            if "datePublished" in line.keys():
                year = line["datePublished"].split("-")[0]
                if year in years:
                    results.append(line["id"])

    def _process_line_chapters_title(self, line, results):
        if "name" in line.keys():
            if "datePublished" in line.keys():
                year = line["datePublished"].split("-")[0]
                if year in years:
                    results[line["id"]] = line["name"]

    def _process_line_chapters_year(self, line, results):
        if "datePublished" in line.keys():
            year = line["datePublished"].split("-")[0]
            if year in years:
                results[line["id"]] = year

    def _process_line_chapters_language(self, line, results):
        if "inLanguage" in line.keys():
            if "datePublished" in line.keys():
                year = line["datePublished"].split("-")[0]
                if year in years:
                    results[line["id"]] = line["inLanguage"]

    def _process_line_chapters_abstract(self, line, results):
        if "description" in line.keys():
            if "datePublished" in line.keys():
                year = line["datePublished"].split("-")[0]
                if year in years:
                    results[line["id"]] = line["description"]

    def _process_line_chapters_authors(self, line, results):
        if "author" in line.keys():
            if "datePublished" in line.keys():
                year = line["datePublished"].split("-")[0]
                if year in years:
                    authors = line["author"]
                    authors_id = [authors[i]["id"] for i in range(
                            len(authors))]
                    results[line["id"]] = authors_id

    def _process_line_chapters_citations(self, line, results):
        if "citation" in line.keys():
            if "datePublished" in line.keys():
                year = line["datePublished"].split("-")[0]
                if year in years:
                    citations = line["citation"]
                    citations_id = [citations[i]["id"] for i in range(
                            len(citations))]
                    results[line["id"]] = citations_id

    def _process_line_chapters_keywords(self, line, results):
        if "keywords" in line.keys():
            if "datePublished" in line.keys():
                year = line["datePublished"].split("-")[0]
                if year in years:
                    results[line["id"]] = line["keywords"]

    def _process_line_chapters_books(self, line, results):
        if "isPartOf" in line.keys():
            if "datePublished" in line.keys():
                year = line["datePublished"].split("-")[0]
                if year in years:
                    results[line["id"]] = line["isPartOf"]["isbn"]
