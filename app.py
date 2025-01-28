from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Setup application
app = Flask(__name__)

# Function to load the trained model and make predictions
def prediction(lst):
    filename = 'model.pkl'  # Path to your saved model
    with open(filename, 'rb') as file:
        model = pickle.load(file)  # Load the model from the pickle file
    pred_value = model.predict([lst])  # Make a prediction
    return pred_value

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pred_value = 0  # Default prediction value
    if request.method == 'POST':
        # Retrieve input values from the form
        house_features = {
            'LotArea': float(request.form['LotArea']),
            'YearBuilt': int(request.form['YearBuilt']),
            '1stFlrSF': float(request.form['1stFlrSF']),
            '2ndFlrSF': float(request.form['2ndFlrSF']),
            'FullBath': int(request.form['FullBath']),
            'BedroomAbvGr': int(request.form['BedroomAbvGr']),
            'TotRmsAbvGrd': int(request.form['TotRmsAbvGrd']),
            'GarageCars': int(request.form['GarageCars']),
            'OverallQual': int(request.form['OverallQual']),
            'OverallCond': int(request.form['OverallCond']),
            'Neighborhood': request.form['Neighborhood'],
            'ExterQual': request.form['ExterQual'],
            'KitchenQual': request.form['KitchenQual'],
            'GrLivArea': float(request.form['GrLivArea'])
        }

        # Convert house features dictionary to a DataFrame
        house_df = pd.DataFrame([house_features])

        # Get the preprocessor from the model pipeline
        filename = 'model.pkl'  # Path to your saved model
        with open(filename, 'rb') as file:
            best_model = pickle.load(file)
        
        preprocessor = best_model.named_steps['preprocessor']

        # Transform the input house data
        house_processed = preprocessor.transform(house_df)

        # Get the model (RandomForestRegressor or XGBRegressor) from the pipeline
        model = best_model.named_steps['model']

        # Predict the house price
        predicted_price = model.predict(house_processed)

        # Display the predicted price
        return render_template('index.html', prediction=f"The predicted sale price for the house is: ${predicted_price[0]:,.2f}")

    return render_template('index.html', pred_value=pred_value)  # Default page load with no prediction

if __name__ == '__main__':
    app.run(debug=True)  # Run the app in debug mode
