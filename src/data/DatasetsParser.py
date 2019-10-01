# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:03:29 2019

@author: Andreea
"""

import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from FileParser import FileParser

import sys
sys.path.insert(0, os.path.join(os.path.realpath(__file__), "..", "..",
                                "utils"))
from TimerCounter import Timer


class DatasetsParser:
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "..", "..", "data", "interim", "parsed_data")

    def __init__(self):
        self.parser = FileParser()
        self.persistent = {}
        self.timer = Timer()
        self.processes = {
                "chapters_books": {
                        "process_data": "_process_data_chapters_books",
                        "persistent_file": os.path.join(
                                self.path, "chapters_books.pkl")
                        },
                "chapters_scigraph_citations": {
                        "process_data": "_process_data_chapters_scigraph_citations",
                        "persistent_file": os.path.join(
                                self.path, "chapters_scigraph_citations.pkl")
                        },
                "books_conferences": {
                        "process_data": "_process_data_books_conferences",
                        "persistent_file": os.path.join(
                                self.path, "books_conferences.pkl")
                        }
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

        # Process the data
        self.persistent[process] = self._parse_file(
                self.processes[process]["process_data"])

        with open(self.processes[process]["persistent_file"], "wb") as f:
            pickle.dump(self.persistent[process], f)

        return self.persistent[process]

    def _parse_file(self, process_data):
        print("Start processing file.\n")
        self.timer.tic()
        process_data_function = self.__getattribute__(process_data)
        results = process_data_function()
        self.timer.toc()
        return results

    # processes implementation
    def _process_data_chapters_books(self):
        # Load datasets
        df_chapters_books_isbns = pd.DataFrame(
                list(self.parser.get_data("chapters_books_isbns").items()),
                columns=["chapter", "books_isbns"])
        df_isbn_book_ids = pd.DataFrame(
                list(self.parser.get_data("isbn_books").items()),
                columns=["isbn", "book"])

        # Process datasets
        df_chapters_books_isbns[["isbn1", "isbn2"]] = pd.DataFrame(
                df_chapters_books_isbns["books_isbns"].tolist(),
                index=df_chapters_books_isbns.index)
        df_chapters_books_isbns.drop(columns=["books_isbns"], axis=1,
                                     inplace=True)
        df_chapters_isbn1 = pd.merge(
                df_chapters_books_isbns[["chapter", "isbn1"]],
                df_isbn_book_ids, how="inner",
                left_on=["isbn1"], right_on=["isbn"])
        df_chapters_isbn1.drop(columns=["isbn1", "isbn"], inplace=True)
        df_chapters_isbn2 = pd.merge(
                df_chapters_books_isbns[["chapter", "isbn2"]],
                df_isbn_book_ids, how="inner",
                left_on=["isbn2"], right_on=["isbn"])
        df_chapters_isbn2.drop(columns=["isbn2", "isbn"], inplace=True)
        df_chapters_books = df_chapters_isbn1.append(df_chapters_isbn2,
                                                     ignore_index=True)
        df_chapters_books.drop_duplicates(inplace=True)
        return df_chapters_books

    def _process_data_chapters_scigraph_citations(self):
        df_chapters_citations = pd.DataFrame(
                list(self.parser.get_data("chapters_citations").items()),
                columns=["chapter", "citations"]
                )
        chapters_count = len(df_chapters_citations)
        with tqdm(desc="Processing citations", total=chapters_count,
                  unit="chapter") as pbar:
            for idx in range(chapters_count):
                citations = df_chapters_citations.iloc[idx]["citations"]
                citations = [c for c in citations if c is not None
                             and c.startswith("sg")]
                df_chapters_citations.iloc[idx]["citations"] = citations if \
                    citations else np.nan
                pbar.update(1)
        return df_chapters_citations[
                df_chapters_citations["citations"].notnull()]

    def _process_data_books_conferences(self):
        df_old_books_new_books = pd.DataFrame(
                    list(self.parser.get_data("old_books_new_books").items()),
                    columns=["old_book", "new_book"])
        df_old_books_conferences = pd.DataFrame(
                    list(self.parser.get_data(
                            "old_books_conferences").items()),
                    columns=["old_book", "conference"])
        df = pd.merge(df_old_books_new_books, df_old_books_conferences,
                      how="left", on=["old_book", "old_book"])
        df.drop(columns=["old_book"], axis=1, inplace=True)
        df.rename(columns={"new_book": "book", "conference": "conference"},
                  inplace=True)
        return df[df["conference"].notnull()]
