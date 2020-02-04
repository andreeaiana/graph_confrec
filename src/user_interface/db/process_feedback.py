# -*- coding: utf-8 -*-
import os
import pickle
import sqlite3
import numpy as np
import pandas as pd


class FeedbackProcessor:

    def __init__(self):
        self.processed_file = "processed_feedback.pkl"

    def process(self):
        if not self._load_feedback():
            print("Feedback not processed. Processing now...")
            feedback = pd.read_sql_table(
                    "feedback",
                    "sqlite:///" + os.path.join(os.path.abspath(
                            os.path.dirname(__file__)), "feedback.db"))
            processed_feedback = list()
            for idx in range(len(feedback)):
                line = feedback.iloc[idx]
                model = line.model_name
                recommendation = self._get_recommendation(line.recommendation)
                confidence = self._get_confidence(line.confidence)
                score = int(line.score)
                comment = line.comment
                if model == "authors":
                    authors = line.input_text.split("Authors: ")[-1].split(
                            "; ")
                    authors = [author for author in authors if author is not ""
                               ]
                    citations = np.NaN
                    abstract = np.NaN
                    title = np.NaN
                else:
                    authors = self._get_authors(line.input_text)
                    citations = self._get_citations(line.input_text)
                    abstract = self._get_abstract(line.input_text)
                    title = self._get_title(line.input_text)
                processed_feedback.append((line.id, model, authors, citations,
                                           title, abstract, recommendation,
                                           confidence, score, comment))

            self.processed_feedback_df = pd.DataFrame(
                    processed_feedback,
                    columns=["id", "model", "authors", "citations", "title",
                             "abstract", "recommendation", "confidence",
                             "score", "comment"])
            self._save_feedback()
            print("Processed.")
        else:
            print("feedback already processed.")

    def _get_recommendation(self, line):
        recommendation = line.split("; ")
        recommendation = [r.lstrip() for r in recommendation]
        return recommendation

    def _get_confidence(self, line):
        confidences = line.split(",")
        confidences = [float(c) for c in confidences]
        return confidences

    def _get_authors(self, line):
        authors = line.split("Authors: ")[-1].split("Citations: ")[0].split(
                "; ")
        authors = [author for author in authors if author is not ""]
        return authors

    def _get_citations(self, line):
        citations = line.split("Citations: ")[-1].split(
                "Abstract: ")[0].split("; ")
        citations = [citation for citation in citations if citation is not ""]
        return citations

    def _get_abstract(self, line):
        return line.split("Abstract: ")[-1].split("Title: ")[0]

    def _get_title(self, line):
        return line.split("Title: ")[-1]

    def _load_feedback(self):
        if os.path.isfile(self.processed_file):
            print("Loading processed feedback...")
            with open(self.processed_feedback_df, "rb") as f:
                self.processed_feedback_df = pickle.load(f)
                print("Loaded.")
            return True
        return False

    def _save_feedback(self):
        print("Saving processed feedback to disk...")
        with open(self.processed_file, "wb") as f:
            pickle.dump(self.processed_feedback_df, f)
        self.processed_feedback_df.to_csv("processed_feedback.csv",
                                          index=False)
        print("Saved.")

    def main():
        print("Starting...")
        from process_feedback import FeedbackProcessor
        processor = FeedbackProcessor()
        processor.process()
        print("Finished.")

    if __name__ == "__main__":
        main()
