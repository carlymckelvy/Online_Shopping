import os
import pandas as pd 
import numpy as np 
import flask
import pickle
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app=Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/activity")
def activity():
    return render_template("activity.html")

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,23)
    model = load_model("model_trained.h5")
    result = model.predict(to_predict)
    return result[0]

@app.route("/predict",methods = ["POST"])
def result():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
    result = ValuePredictor(to_predict_list)
    prediction = str(result)
    return render_template('predict.html',prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)