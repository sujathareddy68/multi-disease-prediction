from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO)


# Now, you can use keras.models instead of tensorflow.keras.models
model = keras.models.Sequential()

app = Flask(__name__)

def predict(values, dic):
    if len(values) == 8:
        model = pickle.load(open('diabetes.pkl','rb'))
        print(type(model))
        values = np.asarray(values)
        return model.predict(values.reshape(1,-1))[0]
    
    elif len(values) == 26:
        model = pickle.load(open('breast-cancer.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1,-1))[0]
    elif len(values) == 13:
        model = pickle.load(open('heart.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1,-1))[0]
    elif len(values) == 18:
        model = pickle.load(open('kidney.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1,-1))[0]
    elif len(values) == 10:
        model = pickle.load(open('liver.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1,-1))[0]

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/diabetes", methods=['GET','POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET','POST'])
def cancerPage():
    return render_template('breast-cancer.html')

@app.route("/heart", methods=['GET','POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET','POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET','POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    if request.method == 'POST':
        try:
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
            return render_template('predict.html', pred=pred)
        except Exception as e:
            logging.error("An error occurred: %s", str(e))
            message = "An error occurred. Please check the logs for details."
            return render_template('predict.html', message=message)
    else:
        # Handle GET request if needed (e.g., when first accessing the page)
        return render_template('predict.html')

if __name__ == "__main__":
    app.run(host="127.0.0.1",port=5000,debug = True)
