# -*- coding: utf-8 -*-
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

base_dir = os.path.abspath(os.path.dirname(__file__))
engine = create_engine("sqlite:///" + os.path.join(base_dir, "feedback.db"),
                       echo=True)
Base = declarative_base()


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    model_name = Column(String)
    input_text = Column(String)
    recommendation = Column(String)
    confidence = Column(String)
    score = Column(String)
    comment = Column(String)

    def __repr__(self):
        return "<Model {}>".format(self.model_name)


Base.metadata.create_all(engine)
