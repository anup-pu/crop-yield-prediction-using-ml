from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests

app = Flask(__name__)

# Load the trained model
model = joblib.load('crop_yield_prediction_model.pkl')

# Replace with your actual API keys and endpoints
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
WEATHER_API_KEY = "4f61e5ac07b86ae596fa8376d363591c"

def get_real_time_data(location):
    weather_params = {
        'q': location,
        'appid': WEATHER_API_KEY,
        'units': 'metric'
    }
    weather_response = requests.get(WEATHER_API_URL, params=weather_params)
    weather_data = weather_response.json()
    
    # Fetch soil data from the CSV file
    soil_data = pd.read_csv('Crop_recommendation.csv')
    
    # Use a placeholder for soil data selection
    selected_soil_data = soil_data.iloc[0]  
    
    combined_data = {
        'N': selected_soil_data['N'],
        'P': selected_soil_data['P'],
        'K': selected_soil_data['K'],
        'temperature': weather_data['main']['temp'],
        'humidity': weather_data['main']['humidity'],
        'ph': selected_soil_data['ph'],
        'rainfall': weather_data.get('rain', {}).get('1h', 0)
    }
    
    return combined_data

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    crop = data['crop']
    season = data['season']
    area = data['area']
    location = data['location']
    
    # Fetch real-time data
    location_data = get_real_time_data(location)
    
    # Load the label encoder if available
    try:
        season_encoder = joblib.load('season_label_encoder.pkl')
    except FileNotFoundError:
        return jsonify({'error': "Label encoder for 'Season' not found. Ensure 'season_label_encoder.pkl' is available."}), 500
    
    # Encode the season
    try:
        season_encoded = season_encoder.transform([season.strip()])[0]
    except ValueError as e:
        return jsonify({'error': f"Error encoding season: {e}"}), 400
    
    # Create input data frame
    input_data = pd.DataFrame([{
        'Area': area,
        'N': location_data['N'],
        'P': location_data['P'],
        'K': location_data['K'],
        'temperature': location_data['temperature'],
        'humidity': location_data['humidity'],
        'ph': location_data['ph'],
        'rainfall': location_data['rainfall'],
        'Season': season_encoded
    }])
    
    # Predict
    predicted_yield = model.predict(input_data)
    
    return jsonify({'predicted_yield': predicted_yield[0]})

if __name__ == "__main__":
    app.run(debug=True)
