from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import joblib
import json

app = Flask(__name__)
heuristic_model = joblib.load('heuristic_model.h5')
rf_model = joblib.load('rf_clf_model.h5')
lr_model = joblib.load('lr_clf_model.h5')
nn_model = joblib.load('nn_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_features = np.array(data['input_features'])

    model_choice = data['model_choice']
    if model_choice == 'heuristic':
        model = heuristic_model
    elif model_choice == 'rf_model':
        model = rf_model
    elif model_choice == 'lr_model':
        model = lr_model
    elif model_choice == 'nn_model':
        model = nn_model

    prediction = model.predict(input_features.reshape(1, -1))[0]

    response = {
        'prediction': int(prediction)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()