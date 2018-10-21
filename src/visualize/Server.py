from flask import Flask, render_template, request
from config import Config
from flask_sqlalchemy import SQLAlchemy
import json
from ModelLoader import ModelLoader

print("Initializing server, please wait until loading/training of models is finished")
print("This may take a while if you do not have any pretrained models on disk")
modelLoader = ModelLoader()

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

from db.dbmodel import Feedback

@app.route("/")
def main():
    return render_template("index.html", modelType="Authors", models=modelLoader.getModels())

@app.route("/auto")
def autocomplete():
    term = request.args.get("term")
    modelName = request.args.get("model")
    auto = modelLoader.autocomplete(modelName, term)
    auto = json.dumps(list(auto))
    auto = bytearray(auto, "utf-8")
    return auto

@app.route("/setModel")
def setModel():
    modelName = request.args.get("model")
    print(modelName)
    if modelName=="Authors":
        modelType = "Authors"
    elif modelName=="Tags":
        modelType = "Tags"
    else:
        modelType = "Abstracts"
    return render_template("input.html", modelType=modelType)

@app.route("/recommend_auto")
def recommend_auto():
    modelName = request.args.get("model")
    data = request.args.get("data")
    query = data.split("; ")
    print(query)
    recommendation = modelLoader.query(modelName, query)
    print(recommendation[0], recommendation[1])
    return render_template("result.html", recommendation=recommendation, feedback_enabled=False)

@app.route("/recommend_abstract")
def recommend_abstract():
    modelName = request.args.get("model")
    data = request.args.get("data")
    print(data)
    recommendation = modelLoader.query(modelName, data)
    print(recommendation[0], recommendation[1])
    return render_template("result.html", recommendation=recommendation, feedback_enabled=False)

@app.route("/feedback")
def feedback():
    modelName = request.args.get("model")
    inputText = request.args.get("inputText")
    recommendation = request.args.get("recommendation")
    confidence = request.args.get("confidence")
    score = request.args.get("score")
    comment = request.args.get("comment")
    #Save it to the DB
    feedback = Feedback(modelName=modelName, inputText=inputText, recommendation=recommendation, 
                        confidence=confidence, score=score, comment=comment)
    db.session.add(feedback)
    try:
        db.session.commit()
        print("Feedback saved to db")
        return render_template("feedback.html", success=True)
    except:
        db.session.rollback()
        print("Error while saving")
        return render_template("feedback.html", success=False)


app.run(port=8080)
