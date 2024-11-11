from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import numpy as np

app = Flask(__name__)

# Load data
data = pd.read_csv("C:\\Users\\User\\Desktop\\ML-car-main\\ML-car-main\\ML\\CarPrice_Assignment.csv")

# Select features and target variable
X = data[['peakrpm', 'horsepower', 'carwidth', 'enginesize', 'stroke', 'citympg', 'highwaympg', 'curbweight']]
y = data['price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing pipeline
numeric_features = X.select_dtypes(include=['int64']).columns
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
])

# Fit the preprocessor on the training data
preprocessor.fit(X_train[numeric_features])

# Define the model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('regressor', HistGradientBoostingRegressor())
])

# Measure training time
start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Training Time: {training_time:.2f} seconds")

# Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²) score: {r2:.2f}")

# Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Cross-validated R² scores: {cv_scores}")
print(f"Average R² score: {cv_scores.mean():.2f}")

# Measure prediction time for a single instance
start_time = time.time()
y_pred_single = model.predict([X_test.iloc[0]])  # Predicting on a single test instance
prediction_time = time.time() - start_time
print(f"Prediction Time for one instance: {prediction_time:.5f} seconds")

@app.route('/', methods=['GET', 'POST'])
def index():
    peakrpm = None
    horsepower = None
    price_prediction = None

    if request.method == 'POST':
        # Retrieve input values from the form
        peakrpm = float(request.form.get('RPM'))
        horsepower = float(request.form.get('horse'))
        carwidth = float(request.form.get('width'))
        enginesize = float(request.form.get('size'))
        stroke = float(request.form.get('stroke'))
        citympg = float(request.form.get('city'))
        highwaympg = float(request.form.get('high'))
        curbweight = float(request.form.get('wght'))

        # Make prediction
        input_data = pd.DataFrame([[peakrpm, horsepower, carwidth, enginesize, stroke, citympg, highwaympg, curbweight]],
                                  columns=['peakrpm', 'horsepower', 'carwidth', 'enginesize', 'stroke', 'citympg', 'highwaympg', 'curbweight'])
        price_prediction = round(model.predict(input_data)[0],2)

    return render_template('Inp.html', rpm=peakrpm, horsepower=horsepower, pred=price_prediction)

if __name__ == '__main__':
    app.run(debug=True)
