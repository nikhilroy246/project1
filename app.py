from flask import Flask, request, jsonify
import joblib
import numpy as np

# Create the Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# Define a route to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    # Scale the input data
    scaled_data = scaler.transform([data])
    # Make a prediction
    prediction = model.predict(scaled_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
