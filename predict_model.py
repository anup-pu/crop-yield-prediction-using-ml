import joblib
import pandas as pd
import numpy as np

def predict_crop_yield(input_data):
    # Load the trained model
    model = joblib.load('crop_yield_model.pkl')

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Convert categorical variables into dummy/indicator variables
    input_df = pd.get_dummies(input_df)

    # Ensure that all necessary columns are present (same as during training)
    model_columns = model.feature_names_in_
    missing_cols = set(model_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    # Reorder columns to match model's feature order
    input_df = input_df[model_columns]
    
    # Predict
    log_prediction = model.predict(input_df)
    prediction = np.expm1(log_prediction)  # Convert log prediction back to original scale
    return prediction[0]
