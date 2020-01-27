# -*- coding: utf-8 -*-
import os
base_dir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or \
        "sqlite:///" + os.path.join(base_dir, "db", "feedback.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
