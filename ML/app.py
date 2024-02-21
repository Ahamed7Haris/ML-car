from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load data
data = pd.read_csv("CarPrice_Assignment.csv")

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

# Define the model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('regressor', HistGradientBoostingRegressor())
])

# Train the model
model.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    peakrpm = None
    horsepower = None
    price_prediction = None

    if request.method == 'POST':
        peakrpm =   (request.form.get('RPM'))  # Convert to   
        horsepower =   (request.form.get('horse'))
        carwidth =   (request.form.get('width'))
        enginesize =   (request.form.get('size'))
        stroke =   (request.form.get('stroke'))
        citympg =   (request.form.get('city'))
        highwaympg =   (request.form.get('high'))
        curbweight =   (request.form.get('wght'))
        
        # Make prediction
        input_data = pd.DataFrame([[peakrpm, horsepower, carwidth, enginesize, stroke, citympg, highwaympg, curbweight]],
                                  columns=['peakrpm', 'horsepower', 'carwidth', 'enginesize', 'stroke', 'citympg', 'highwaympg', 'curbweight'])
        price_prediction = model.predict(input_data)[0]

    return render_template('Inp.html', rpm=peakrpm, horsepower=horsepower, pred=price_prediction)

if __name__ == '__main__':
    app.run(debug=True)
