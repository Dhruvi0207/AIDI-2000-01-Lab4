from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model, scaler, and label encoder
model = joblib.load('fish_weight_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        length = float(data['length'])
        height = float(data['height'])
        width = float(data['width'])
        species = label_encoder.transform([data['species']])[0]

        # Since the model was trained with Length1, Length2, Length3
        features = pd.DataFrame([[species, length, length, length, height, width]], 
                                columns=['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width'])
        features = scaler.transform(features)

        # Make prediction
        weight = model.predict(features)[0]

        return jsonify({'weight': weight})
    except (ValueError, KeyError) as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)