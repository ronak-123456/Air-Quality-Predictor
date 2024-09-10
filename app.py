from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load('best_model.joblib')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_aqi():
    # Get the data from the request body
    data = request.json
    
    # Extract features from the input data
    features = [
        data.get('pm25', 0), 
        data.get('pm10', 0), 
        data.get('no2', 0), 
        data.get('co', 0), 
        data.get('so2', 0), 
        data.get('o3', 0), 
        data.get('temperature', 0), 
        data.get('humidity', 0), 
        data.get('wind_speed', 0), 
        data.get('wind_direction', 0)
    ]
    
    # Reshape features for the model
    features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return the predicted AQI
    return jsonify({'aqi': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
