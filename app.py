import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess the data
def load_and_preprocess_data():
    url = 'https://raw.githubusercontent.com/JustGlowing/vgsales/master/vgsales.csv'
    data = pd.read_csv(url)
    data = data.dropna()
    data = pd.get_dummies(data, columns=['Platform', 'Genre', 'Publisher'], drop_first=True)
    return data

# Function to train the model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Load and preprocess data
data = load_and_preprocess_data()

# Define features and target
X = data.drop(columns=['Global_Sales'])
y = data['Global_Sales']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = train_model(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Streamlit app
st.title('Video Game Sales Prediction')

st.write("## Dataset Overview")
st.write(data.head())

st.write("## Model Performance")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"Root Mean Squared Error: {rmse:.2f}")

# Feature input from the user
st.write("## Predict Future Sales")

# Inputs for features
features = {}
for column in X.columns:
    if data[column].dtype == 'object':
        # Handle categorical features
        unique_values = data[column].unique()
        features[column] = st.selectbox(f"Select {column}", unique_values)
    else:
        features[column] = st.number_input(f"Enter {column}", value=float(data[column].mean()))

# Convert input features to DataFrame
input_features = pd.DataFrame([features], columns=X.columns)

# Predict
if st.button('Predict Sales'):
    prediction = model.predict(input_features)
    st.write(f"Predicted Global Sales: ${prediction[0]:.2f}M")

# Optional: Plot feature importances
st.write("## Feature Importances")
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
ax.set_title('Feature Importances')
st.pyplot(fig)
