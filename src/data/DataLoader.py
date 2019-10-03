# -*- coding: utf-8 -*-
import os
import pandas as pd
from FileParser import FileParser
from DatasetsParser import DatasetsParser


class DataLoader:
    """
    Class for loading the data needed by the models.
    Functions return self, so calls can be concatenated.

    Possible functions:
        - papers: adds papers + attributes (book_id, title, language)
        - abstracts: adds abstracts
        - citations: adds citations
        - conferences: adds conferences +
            attributes (acronym, city, country, dateend, datestart, name, year)
        - conferenceseries: adds conferenceseries + attributes (name)

    The range of years needs to be given at an appropriate call:
        papers, abstracts or citations

    Calls are left merges, so the order of the calls matter.

    Examples
    ------------------------------------
        import DataLoader
        d = DataLoader.DataLoader()

        Abstracts in 2016:
            d.abstracts(["2016"])
        All Conferences:
            d.conferences()
        Papers in 2015 with conferences:
            d.papers(["2015"]).conferences()
        Papers in 2015, 2016 with abstracts, citations, conferences and
        conferenceseries
            d.papers(
            ["2015","2016"]
            ).abstracts().contributions().conferences().conferenceseries()
    """

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "..", "..", "data" "processed")

    def __init__(self):
        self.parser = FileParser()
        self.dt_parser = DatasetsParser()

    # Add papers
    def papers(self, years=None):
        if hasattr(self, "years") and years is not None:
            raise AttributeError("Years already set.")
        elif years is not None:
            self.years = years
        elif not hasattr(self, "years"):
            raise AttributeError("Years needed.")

        df_chapters_books = self.dt_parser.get_data("chapters_books")
        df_chapters_books.rename(
                columns={"chapter": "chapter", "book_id": "book"},
                inplace=True)
        df_chapter_title = pd.DataFrame(
                            list(self.parser.get_data(
                                    "chapters_title").items()),
                            columns=["chapter", "chapter_title"])
        df_chapters_language = pd.DataFrame(
                        list(self.parser.get_data(
                                "chapters_language").items()),
                        columns=["chapter", "chapter_language"])
        df_chapter_years = pd.DataFrame(
                        list(self.parser.get_data("chapters_year").items()),
                        columns=["chapter", "chapter_year"])

        df = pd.merge(df_chapters_books, df_chapter_title,
                      how="left", on=["chapter", "chapter"])
        df = pd.merge(df, df_chapters_language,
                      how="left", on=["chapter", "chapter"])
        df = pd.merge(df, df_chapter_years,
                      how="left", on=["chapter", 'chapter'])

        data = df
        data = data[data.chapter_year.isin(self.years)]
        data = data[data.chapter_language == "en"]

        if hasattr(self, "data"):
            if "chapter" in self.data.keys():
                self.data = pd.merge(self.data, data,
                                     how="left", on=["chapter", "chapter"])
            else:
                raise KeyError("Needs papers.")
        else:
            self.data = data

        return self

    # Add abstracts
    def abstracts(self, years=None):
        if hasattr(self, "years") and years is not None:
            raise AttributeError("Years already set.")
        elif years is not None:
            self.years = years
        elif not hasattr(self, "years"):
            raise AttributeError("Years needed.")

        df_chapters_abstract = pd.DataFrame(
                        list(self.parser.get_data(
                                "chapters_abstract").items()),
                        columns=["chapter", "chapter_abstract"])
        data = df_chapters_abstract

        if hasattr(self, "data"):
            if "chapter" in self.data.keys():
                self.data = pd.merge(self.data, data,
                                     how="left", on=["chapter", "chapter"])
            else:
                raise KeyError("Needs papers.")
        else:
            self.data = data

        return self

    # Add citations
    def citations(self, years=None):
        if hasattr(self, "years") and years is not None:
            raise AttributeError("Years already set.")
        elif years is not None:
            self.years = years
        elif not hasattr(self, "years"):
            raise AttributeError("Years needed.")
        df_chapters_citations = self.dt_parser.get_data(
                "chapters_scigraph_citations")
        data = df_chapters_citations

        if hasattr(self, "data"):
            if "chapter" in self.data.keys():
                self.data = pd.merge(self.data, data,
                                     how="left", on=["chapter", "chapter"])
            else:
                raise KeyError("Needs papers.")
        else:
            self.data = data

        return self

    # Add conferences
    def conferences(self):
        if not hasattr(self, "data"):
            df_conferences = pd.DataFrame(self.parser.get_data("conferences"),
                                          columns=["conference"])
        elif "chapter" in self.data.keys():
            df_conferences = self.dt_parser.get_data("books_conferences")
        else:
            raise KeyError("Needs papers.")

        df_conferences_acronym = pd.DataFrame(
                list(self.parser.get_data("conferences_acronym").items()),
                columns=["conference", "conference_acronym"])
        df_conferences_city = pd.DataFrame(
                list(self.parser.get_data("conferences_city").items()),
                columns=["conference", "conference_city"])
        df_conferences_country = pd.DataFrame(
                list(self.parser.get_data("conferences_country").items()),
                columns=["conference", "conference_country"])
        df_conferences_datestart = pd.DataFrame(
                list(self.parser.get_data("conferences_datestart").items()),
                columns=["conference", "conference_datestart"])
        df_conferences_dateend = pd.DataFrame(
                list(self.parser.get_data("conferences_dateend").items()),
                columns=["conference", "conference_dateend"])
        df_conferences_name = pd.DataFrame(
                list(self.parser.get_data("conferences_name").items()),
                columns=["conference", "conference_name"])
        df_conferences_year = pd.DataFrame(
                list(self.parser.get_data("conferences_year").items()),
                columns=["conference", "conference_year"])

        df = pd.merge(df_conferences, df_conferences_acronym,
                      how="left", on=["conference", "conference"])
        df = pd.merge(df, df_conferences_city, how="left",
                      on=["conference", "conference"])
        df = pd.merge(df, df_conferences_country, how="left",
                      on=["conference", "conference"])
        df = pd.merge(df, df_conferences_dateend, how="left",
                      on=["conference", "conference"])
        df = pd.merge(df, df_conferences_datestart, how="left",
                      on=["conference", "conference"])
        df = pd.merge(df, df_conferences_name, how="left",
                      on=["conference", "conference"])
        df = pd.merge(df, df_conferences_year, how="left",
                      on=["conference", "conference"])

        df.conference_acronym = df.conference_acronym.str[1:-1]
        df.conference_name = df.conference_name.str[1:-1]
        df.conference_city = df.conference_city.str[1:-1]
        df.conference_country = df.conference_country.str[1:-1]
        df.conference_datestart = df.conference_datestart.str[1:11]
        df.conference_dateend = df.conference_dateend.str[1:11]
        df.conference_year = df.conference_year.str[1:5]

        if hasattr(self, "data"):
            self.data = pd.merge(self.data, df, how="left",
                                 on=["book", "book"])
        else:
            self.data = df

        return self

    # Add conferenceseries
    def conferenceseries(self):
        if not hasattr(self, "data"):
            df_conferenceseries = pd.DataFrame(
                    self.parser.get_data("conferenceseries"),
                    columns=["conferenceseries"])
        elif "conference" in self.data.keys():
            df_conferenceseries = pd.DataFrame(
                    list(self.parser.get_data(
                            "conferences_conferenceseries").items()),
                    columns=["conference", "conferenceseries"])
        else:
            raise KeyError("Needs conferences.")

        df_conferenceseries_name = pd.DataFrame(
                list(self.parser.get_data("conferenceseries_name").items()),
                columns=["conferenceseries", "conferenceseries_name"])
        df = pd.merge(df_conferenceseries, df_conferenceseries_name,
                      how="left", on=["conferenceseries", "conferenceseries"])
        df.conferenceseries_name = df.conferenceseries_name.str[1:-1]

        if hasattr(self, "data"):
            self.data = pd.merge(self.data, df, how="left",
                                 on=["conference", "conference"])
        else:
            self.data = df

        return self

    # Add author ids
    def authors_ids(self, years):
        if hasattr(self, "years") and years is not None:
            raise AttributeError("Years already set.")
        elif years is not None:
            self.years = years
        elif not hasattr(self, "years"):
            raise AttributeError("Years needed.")
        df_author_id_chapters = self.dt_parser.get_data("author_id_chapters")
        data = df_author_id_chapters

        if hasattr(self, "data"):
            if "chapter" in self.data.keys():
                self.data = pd.merge(self.data, data,
                                     how="left", on=["chapter", "chapter"])
            else:
                raise KeyError("Needs papers.")
        else:
            self.data = data

        return self

    # Add author names:
    def author_names(self, years):
        if hasattr(self, "years") and years is not None:
            raise AttributeError("Years already set.")
        elif years is not None:
            self.years = years
        elif not hasattr(self, "years"):
            raise AttributeError("Years needed.")
        df_author_name_chapters = self.dt_parser.get_data(
                "author_name_chapters")
        data = df_author_name_chapters

        if hasattr(self, "data"):
            if "chapter" in self.data.keys():
                self.data = pd.merge(self.data, data,
                                     how="left", on=["chapter", "chapter"])
            else:
                raise KeyError("Needs papers.")
        else:
            self.data = data

        return self

    # Get training data
    def training_data(self, years=None):
        if years is not None:
            years = years
        else:
            years = self.parser.years.copy()
            years.remove("2015")
            years.remove("2016")
        return self.papers(years).conferences().conferenceseries()

    # Get validation data
    def validation_data(self, years=None):
        if years is not None:
            years = years
        else:
            years = ["2015"]
        return self.papers(years).conferences().conferenceseries()

    # Get test data
    def test_data(self, years=None):
        if years is not None:
            years = years
        else:
            years = ["2016"]
        return self.papers(years).conferences().conferenceseries()

    # Get training data with abstracts
    def training_data_with_abstracts(self, years=None):
        self.training_data(years).abstracts()
        self.data = self.data[["chapter_abstract", "conferenceseries",
                               "chapter_title"]].copy()
        self.data.drop(
                list(self.data[pd.isnull(self.data.chapter_abstract)].index),
                inplace=True
                )
        self.data = self.data.reset_index()
        self.data = self.data[["chapter_abstract", "conferenceseries",
                               "chapter_title"]]
        return self

    # Get training data with abstracts and citations
    def training_data_with_abstracts_citations(self, years=None):
        self.training_data(years).abstracts().citations()
        self.data = self.data[["chapter_abstract", "chapter_citations",
                               "chapter_title", "conferenceseries"]].copy()
        self.data.drop(
                list(self.data[pd.isnull(self.data.chapter_abstract)].index),
                inplace=True
                )
        self.data = self.data.reset_index()
        self.data.drop(
                list(self.data[pd.isnull(self.data.chapter_citations)].index),
                inplace=True
                )
        self.data = self.data.reset_index()
        self.data = self.data[["chapter_abstract", "chapter_citations",
                               "chapter_title", "conferenceseries"]]
        return self

    # Get validation data with abstracts
    def validation_data_with_abstracts(self, years=None):
        self.validation_data(years).abstracts()
        self.data = self.data[["chapter_abstract", "conferenceseries",
                               "chapter_title"]].copy()
        self.data.drop(
                list(self.data[pd.isnull(self.data.chapter_abstract)].index),
                inplace=True
                )
        self.data = self.data.reset_index()
        self.data = self.data[["chapter_abstract", "conferenceseries",
                               "chapter_title"]]
        return self

    # Get validation data with abstracts and citations
    def validation_data_with_abstracts_citations(self, years=None):
        self.validation_data(years).abstracts().citations()
        self.data = self.data[["chapter_abstract", "chapter_citations",
                               "chapter_title", "conferenceseries"]].copy()
        self.data.drop(
                list(self.data[pd.isnull(self.data.chapter_abstract)].index),
                inplace=True
                )
        self.data = self.data.reset_index()
        self.data.drop(
                list(self.data[pd.isnull(self.data.chapter_citations)].index),
                inplace=True
                )
        self.data = self.data.reset_index()
        self.data = self.data[["chapter_abstract", "chapter_citations",
                               "chapter_title", "conferenceseries"]]
        return self

    # Get test data with abstracts
    def test_data_with_abstracts(self, years=None):
        self.test_data(years).abstracts()
        self.data = self.data[["chapter_abstract", "conferenceseries",
                               "chapter_title"]].copy()
        self.data.drop(
                list(self.data[pd.isnull(self.data.chapter_abstract)].index),
                inplace=True
                )
        self.data = self.data.reset_index()
        self.data = self.data[["chapter_abstract", "conferenceseries",
                               "chapter_title"]]
        return self

    # Get test data with abstracts and citations
    def test_data_with_abstracts_citations(self, years=None):
        self.test_data(years).abstracts().citations()
        self.data = self.data[["chapter_abstract", "chapter_citations",
                               "chapter_title", "conferenceseries"]].copy()
        self.data.drop(
                list(self.data[pd.isnull(self.data.chapter_abstract)].index),
                inplace=True
                )
        self.data = self.data.reset_index()
        self.data.drop(
                list(self.data[pd.isnull(self.data.chapter_citations)].index),
                inplace=True
                )
        self.data = self.data.reset_index()
        self.data = self.data[["chapter_abstract", "chapter_citations",
                               "chapter_title", "conferenceseries"]]
        return self

    # Get test data with abstracts ready for evaluation
    def evaluation_data_with_abstracts(self, years=None):
        self.test_data_with_abstracts(years)
        quey_test = list(self.data.chapter_abstract)

        conferences_truth = list()
        confidences_truth = list()

        for conference in list(self.data.conferenceseries):
            conferences_truth.append([conference])
            confidences_truth.append([1])

        truth = [conferences_truth, confidences_truth]
        return quey_test, truth

    # Get test data with abstracts and citations ready for evaluation
    def evaluation_data_with_abstracts_citations(self, years=None):
        self.test_data_with_abstracts_citations(years)
        quey_test = list(self.data.chapter_abstract)

        conferences_truth = list()
        confidences_truth = list()

        for conference in list(self.data.conferenceseries):
            conferences_truth.append([conference])
            confidences_truth.append([1])

        truth = [conferences_truth, confidences_truth]
        return quey_test, truth
