from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

def predict_crop_yield(input_data):
    # Load the trained model
    model = joblib.load('crop_yield_model.pkl')

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Convert categorical variables into dummy/indicator variables
    input_df = pd.get_dummies(input_df)

    # Ensure that all necessary columns are present (same as during training)
    model_columns = model.feature_names_in_
    missing_cols = [col for col in model_columns if col not in input_df.columns]
    
    # Add missing columns with default value 0 in one step
    missing_df = pd.DataFrame(0, index=np.arange(len(input_df)), columns=missing_cols)
    input_df = pd.concat([input_df, missing_df], axis=1)
    
    # Reorder columns to match model's feature order
    input_df = input_df[model_columns]

    # Predict
    log_prediction = model.predict(input_df)
    prediction = np.expm1(log_prediction)  # Convert log prediction back to original scale
    
    # Convert prediction to standard float
    return float(prediction[0])

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/learning')
def learning():
    return render_template('learning.html')

@app.route('/financial')
def financial():
    return render_template('financial.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.content_type == 'application/json':
            # JSON data is being received
            input_data = request.json
        else:
            # Form data is being received
            input_data = {
                'State_Name': request.form['State_Name'],
                'District_Name': request.form['District_Name'],
                'Crop_Year': int(request.form['Crop_Year']),
                'Season': request.form['Season'],
                'Crop': request.form['Crop'],
                'Area': float(request.form['Area'])
            }
        
        print(f"Received input data: {input_data}")  # Debug print

        # Predict crop yield
        prediction = predict_crop_yield(input_data)
        
        # Return prediction as JSON
        return jsonify({'prediction': round(prediction, 2)})

    except Exception as e:
        # Log the error and return a JSON error message
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred. Please check your input data.'}), 500


if __name__ == '__main__':
    app.run(debug=True)
