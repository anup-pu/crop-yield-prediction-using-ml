import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('crop_production.csv')

# Fill missing values with the mean of the column
data['Production'] = data['Production'].fillna(data['Production'].mean())

# Optional: Analyze the distribution of the target variable
plt.hist(data['Production'], bins=50)
plt.title('Distribution of Production')
plt.xlabel('Production')
plt.ylabel('Frequency')
plt.show()

# Log transform the target variable to reduce skewness
y = np.log1p(data['Production'])

# Feature selection and preprocessing
X = data[['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop', 'Area']]
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBRegressor
xgb = XGBRegressor(random_state=42, n_jobs=-1)

# Set up the GridSearchCV parameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'learning_rate': [0.01, 0.1, 0.3]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
print("Starting model training...")
grid_search.fit(X_train, y_train)
print("Model training completed.")

# Best model from Grid Search
best_model = grid_search.best_estimator_

# Evaluate the best model using cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-Validated MSE: {-cv_scores.mean()}')

# Evaluate the best model on the test set
predictions = np.expm1(best_model.predict(X_test))
mse = mean_squared_error(np.expm1(y_test), predictions)
print(f'Mean Squared Error: {mse}')

# Optional: Analyze residuals
plt.scatter(predictions, predictions - np.expm1(y_test))
plt.hlines(y=0, xmin=min(predictions), xmax=max(predictions))
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Save the trained model
joblib.dump(best_model, 'crop_yield_model.pkl')
