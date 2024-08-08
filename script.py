import streamlit as st
import pandas as pd
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import time

# Define the paths to your joblib files
joblib_file_path = 'knnmainnew_model.joblib'
label_encoder_path = 'encoder.joblib'
scaler_path = 'scaler.joblib'

# Load the model, label encoder, and scaler using joblib
model = joblib.load(joblib_file_path)
label_encoder = joblib.load(label_encoder_path)
scaler = joblib.load(scaler_path)

st.title("Environmental Monitoring App")

# Initialize session state variables to track previous prediction and plot data
if 'previous_prediction' not in st.session_state:
    st.session_state.previous_prediction = 'M'  # Default value or set to 'unknown'
if 'plot_data' not in st.session_state:
    st.session_state.plot_data = pd.DataFrame(columns=['Week', 'Temperature', 'Humidity', 'GasLevel', 'Predicted Status', 'Timestamp'])

st.write("Real-Time Prediction from Google Sheets:")

# Google Sheets API setup
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

# Load credentials from Streamlit secrets
creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["google_api"], scope)
client = gspread.authorize(creds)

# Open the Google Sheet by URL
sheet_url = 'https://docs.google.com/spreadsheets/d/1lbGCOmPlX4HXzNW2WDfocolRO6E28uFGTNeeH_yBIbo/edit#gid=0'
sheet = client.open_by_url(sheet_url)

# Access specific worksheet (optional)
worksheet = sheet.get_worksheet(0)  # Index of worksheet, 0 for the first sheet

def fetch_and_predict():
    # Read data from the sheet
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])

    # Ensure Week, Temp, Hum, and Gas columns exist and exclude Date and Time
    if all(col in df.columns for col in ['Week', 'Temp', 'Hum', 'Gas']):
        # Convert columns to appropriate data types
        df['Week'] = pd.to_numeric(df['Week'], errors='coerce')
        df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
        df['Hum'] = pd.to_numeric(df['Hum'], errors='coerce')
        df['Gas'] = pd.to_numeric(df['Gas'], errors='coerce')
        
        # Drop rows with any NaN values
        df.dropna(subset=['Week', 'Temp', 'Hum', 'Gas'], inplace=True)

        # Select the required columns and scale them
        features = df[['Week', 'Temp', 'Hum', 'Gas']]
        features_scaled = scaler.transform(features)

        # Predict using the model
        predictions = model.predict(features_scaled)

        # Add predictions to the DataFrame
        df['Prediction'] = predictions
        df['Timestamp'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])

        # Update session state plot data
        st.session_state.plot_data = pd.concat([st.session_state.plot_data, df[['Week', 'Temp', 'Hum', 'Gas', 'Prediction', 'Timestamp']]])

        # Map prediction results to colors
        color_map = {'S': 'green', 'M': 'orange', 'U': 'red'}
        df['Color'] = df['Prediction'].map(color_map)

        # Plot results
        fig, ax = plt.subplots()
        for label, color in color_map.items():
            subset = df[df['Prediction'] == label]
            ax.scatter(subset['Timestamp'], [label] * len(subset), color=color, label=label)

        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Prediction')
        ax.set_title('Real-Time Prediction Results')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        st.pyplot(fig)

if st.button('Fetch and Predict'):
    fetch_and_predict()

# Continuous updating
st.write("This section will refresh every 60 seconds to fetch new data and update predictions.")
while True:
    fetch_and_predict()
    time.sleep(60)
