from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
import os
basedir = os.path.abspath(os.path.dirname(__file__))

engine = create_engine('sqlite:///' + os.path.join(basedir, "feedback.db"), echo=True)
Base = declarative_base()


class Feedback(Base):
    
    __tablename__= "feedback"
    id = Column(Integer, primary_key=True)
    modelName = Column(String)
    inputText = Column(String)
    recommendation = Column(String)
    confidence = Column(String)
    score = Column(Integer)
    comment = Column(String)

    def __repr__(self):
        return '<Model {}>'.format(self.modelName)    


Base.metadata.create_all(engine)