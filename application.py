from dask.array.random import random_integers
from flask import Flask, request,render_template
import numpy as np
import pandas as pd
import sklearn
import pickle
import warnings


# Importing Model of pkl extension maked when running model in jupyter notebook

model = pickle.load(open('recommendation_model.pkl','rb'))

# Creating App by using Flask
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosphorus'])
    K = int(request.form['Potassium'])
    temperature = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N,P,K,temperature,humidity,ph,rainfall]
    single_prediction = np.array(feature_list).reshape(1,-1)

    prediction = model.predict(single_prediction)

    label_dict = {
        1: 'rice',
        2: 'maize',
        3: 'jute',
        4: 'cotton',
        5: 'coconut',
        6: 'papaya',
        7: 'orange',
        8: 'apple',
        9: 'muskmelon',
        10: 'watermelon',
        11: 'grapes',
        12: 'mango',
        13: 'banana',
        14: 'pomegranate',
        15: 'lentil',
        16: 'blackgram',
        17: 'mungbean',
        18: 'mothbeans',
        19: 'pigeonpeas',
        20: 'kidneybeans',
        21: 'chickpea',
        22: 'coffee'
    }

    if prediction[0] in label_dict:
        crop = label_dict[prediction[0]]
        result = "{}.".format(crop)
    else:
        result = "Sorry, Not Able to Recommend What to Cultivate"
    return render_template('index.html',result = result)

# Main Function in Python
if __name__ == '__main__':
    app.run(debug=True)