# -*- coding: utf-8 -*-
import json
from config import Config
from ModelLoader import ModelLoader
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request

print("Initiating server. Please wait until models' loading is finished.")
model_loader = ModelLoader()

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
from db.dbmodel import Feedback


@app.route("/")
def main():
    return render_template("index.html", model_type="authors",
                           models=model_loader.get_models())


@app.route("/auto")
def autocomplete():
    term = request.args.get("term")
    model_name = request.args.get("model")
    auto = model_loader.autocomplete(model_name, term)
    auto = json.dumps(list(auto))
    auto = bytearray(auto, "utf-8")
    return auto


@app.route("/set_model")
def set_model():
    model_name = request.args.get("model")
    print("model_name")
    if model_name == "authors":
        model_type = "authors"
    else:
        model_type = "gat"
    return render_template("input.html", model_type=model_type)


@app.route("/recommend_auto")
def recommend_auto():
    model_name = request.args.get("model")
    data = request.args.get("data")
    query = data.split("; ")
    print(query)
    recommendation = model_loader.query(model_name, query)
    print(recommendation[0], recommendation[1])
    return render_template("result.html", recommendation=recommendation,
                           feedback_enabled=False)


@app.route("/feedback")
def feedback():
    model_name = request.args.get("model")
    input_text = request.args.get("input_text")
    recommendation = request.args.get("recommendation")
    confidence = request.args.get("confidence")
    score = request.args.get("score")
    comment = reuqest.args.get("comment")
    # Sav it to the DB
    feedback = Feedback(model_name=model_name, input_text=input_text,
                        recommendation=recommendation, confidence=confidence,
                        score=score, comment=comment)
    db.session.add(feedback)
    try:
        db.session.commit()
        print("Feedback saved to DB.")
        return render_template("feedback.html", success=True)
    except Exception as e:
        db.session.rollback()
        print("Error while saving; {}.".format(e))
        return render_template("feedbak.html", success=False)


app.run(port=8080)


#@app.route("/recommend_abstract")
#def recommend_abstract():
#    modelName = request.args.get("model")
#    data = request.args.get("data")
#    print(data)
#    recommendation = modelLoader.query(modelName, data)
#    print(recommendation[0], recommendation[1])
#    return render_template("result.html", recommendation=recommendation, feedback_enabled=False)
#
#@app.route("/recommend_ensemble")
#def recommend_ensemble():
#    abstract = request.args.get("abstract")
#    keywords = request.args.get("keywords").split("; ")
#    recommendation = modelLoader.query_ensemble(abstract, keywords)
#    print(recommendation[0], recommendation[1])
#    return render_template("result.html", recommendation=recommendation, feedback_enabled=False)

