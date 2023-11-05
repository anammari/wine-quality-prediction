from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

with open("./encoder.pkl", "rb") as infile:
    encoder = joblib.load(infile)

with open("./best_model.pkl", "rb") as infile:
    best_model = joblib.load(infile)

def prepare_features(obs):
    obs_df = pd.DataFrame([obs])
    client_data = encoder.transform(obs_df)
    return client_data

def predict(client_data):
    pred = best_model.predict(client_data)
    return pred

app = Flask('wine-quality-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    obs = request.get_json()

    client_data = prepare_features(obs)
    pred = predict(client_data)

    data = {'score': round(pred[0], 2)}
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)