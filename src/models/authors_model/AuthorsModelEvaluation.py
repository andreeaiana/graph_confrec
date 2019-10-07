# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "evaluations"))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "data"))
from DataLoader import DataLoader
from AuthorsModel import AuthorsModel
from EvaluationContainer import EvaluationContainer

# Load training data and train model
print("Training the model.")
d = DataLoader()
d.training_data().author_names()
model = AuthorsModel(rec=10)
model.train(d.data)
print("Model trained.\n")

# Generate validation data
d = DataLoader()
d.validation_data().author_names()
d.data.author_name = d.data.author_name.str.decode(
        "unicode_escape").str.lower()

conferenceseries = d.data[[
        "chapter", "conferenceseries"]].copy().drop_duplicates().reset_index()
authors = d.data.groupby("chapter")["author_name"].apply(list)

data_validation = conferenceseries.join(authors, on="chapter")
query = list(data_validation["author_name"])
truth = [list(data_validation["conferenceseries"].apply(lambda x: [x]))]

print("Getting recommendations.")
recommendations = model.query_batch(query)
print("Recommendations computed.\n")

# Evaluate the model on validation data
print("Evaluating the model.")
evaluator = EvaluationContainer()
evaluator.evaluate(recommendations, truth)
print("Model evaluated.\n")
count_none = sum(x is None for x in recommendations[0])
size_query = len(query)
print("{}/{} recommendations are None = {}%.".format(count_none, size_query,
      count_none/size_query*100))

# Generate test data
d = DataLoader()
d.test_data().author_names()
d.data.author_name = d.data.author_name.str.decode(
        "unicode_escape").str.lower()

conferenceseries = d.data[[
        "chapter", "conferenceseries"]].copy().drop_duplicates().reset_index()
authors = d.data.groupby("chapter")["author_name"].apply(list)

data_test = conferenceseries.join(authors, on="chapter")
query = list(data_test["author_name"])
truth = [list(data_test["conferenceseries"].apply(lambda x: [x]))]

print("Getting recommendations.")
recommendations = model.query_batch(query)
print("Recommendations computed.\n")

# Evaluate the model on test data
print("Evaluating the model.")
evaluator = EvaluationContainer()
evaluator.evaluate(recommendations, truth)
print("Model evaluated.\n")
count_none = sum(x is None for x in recommendations[0])
size_query = len(query)
print("{}/{} recommendations are None = {}%.".format(count_none, size_query,
      count_none/size_query*100))
