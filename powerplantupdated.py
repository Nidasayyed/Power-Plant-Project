# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:27:35 2024

@author: Dell
"""
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Function to load the model
#st.title('Power Plant: Random Forest')
#st.sidebar.header('User Input Parameters')

data = pd.read_csv("/Users/vidya/Downloads/Deployment(1)/Copy of energy_production (1).csv")

# Separate features and target variable
X = data[['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity']]
y = data['energy_production']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Save the model to disk
joblib.dump(rf_model, 'random_forest_model.pkl')

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

# Function to make predictions
def predict_energy(model, temperature, exhaust_vacuum, amb_pressure, r_humidity):
    prediction = model.predict([[temperature, exhaust_vacuum, amb_pressure, r_humidity]])
    return prediction[0]



# Function to get user input
def get_user_input():
    temperature = st.text_input('Temperature')
    exhaust_vacuum = st.text_input('Exhaust Vacuum')
    amb_pressure = st.text_input('Ambient Pressure')
    r_humidity = st.text_input('Relative Humidity')
    return temperature, exhaust_vacuum, amb_pressure, r_humidity

# Main code starts here
if __name__ == "__main__":
    # Set title and sidebar header 
    st.title('Power Plant: Random Forest')
    st.subheader('User Input Parameters')
    
    # Get user input
    Temperature, Exhaust_Vacuum, Ambient_Pressure, Relative_Humidity = get_user_input()
    
    # Make prediction when the user clicks the button
    if st.button('Predict'):
        prediction = predict_energy(model, Temperature, Exhaust_Vacuum, Ambient_Pressure, Relative_Humidity)
        st.subheader(f'Predicted Energy Output: {prediction:.2f} MW')

