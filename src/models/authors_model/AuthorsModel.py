# -*- coding: utf-8 -*-


from AbstractClasses import AbstractModel


class AuthorsModels(AbstractModel):

    def __init__(self, rec=10):
        self.rec = rec

    def query_single(self, authors):
        """Queries the model and returns a list of recommendations.

        Args:
            authors (str[]): A list containing author names.

        Returns:
            str[]: name of the conference.
            double[]: confidence scores
        """
        return self.query_batch([authors])[0]

    def query_batch(self, batch):
        """Queries the model and returns a list of recommendations for each
        request.

        Args:
            batch[str[]]: A list containing lists of author names.

        Returns:
            A list of size 'len(batch)' which contains the recommendations for
            each item of the batch. If author not found, the value is None.

            str[]: name of the conference.
            double[]: confidence scores.
        """
        if not isinstance(batch, list):
            raise TypeError("Argument 'batch' needs to be a list of lists\
                            containing author names.")
        conference = list()
        confidence = list()

        for index, authors in enumerate(batch):
            result = self.data[
                    self.data["author_name"].isin(authors)
                    ][["conferenceseries", "count"]].groupby(
                    "conferenceseries").sum().sort_values(
                            by="count", ascending=False)[0:self.rec]
            if len(result) == 0:
                conference.append(None)
                confidence.append(None)
            else:
                conference.append(list(result.index))
                confidence.append(list(result["count"]))

        return [conference, confidence]

    def train(self, data):
        """Set the data to be searched for by author name.
        Needs to contain 'author_name' and 'conferenceseries'.

        Args:
            data (pandas.DataFrame): the data used by the model.
        """
#        AbstractModel.train(self, data)
        for check in ["author_name", "conferenceseries"]:
            if check not in data.columns:
                raise IndexError(
                        "Column '{}' not contained in given DataFrame.".format(
                                check))
        data.author_name = data.author_name.str.decode(
                "unicode_escape").str.lower()
        data["count"] = 0
        self.data = data.groupby(
                ["author_name", "conferenceseries"]).count().reset_index()[
                ["author_name", "conferenceseries", "count"]]
