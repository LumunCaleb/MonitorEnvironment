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

# Initialize session state variables
if 'previous_prediction' not in st.session_state:
    st.session_state.previous_prediction = 'M'
if 'plot_data' not in st.session_state:
    st.session_state.plot_data = pd.DataFrame(columns=['Week', 'Temp', 'Hum', 'Gas', 'Prev_Status', 'Prediction', 'Timestamp'])

st.write("Real-Time Prediction from Google Sheets:")

# Google Sheets API setup
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["google_api"], scope)
client = gspread.authorize(creds)
sheet_url = 'https://docs.google.com/spreadsheets/d/1lbGCOmPlX4HXzNW2WDfocolRO6E28uFGTNeeH_yBIbo/edit#gid=0'
sheet = client.open_by_url(sheet_url)
worksheet = sheet.get_worksheet(0)

def fetch_and_predict():
    # Read data from the sheet
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])

    # Ensure required columns exist
    expected_columns = ['Week', 'Temp', 'Hum', 'Gas', 'Prev. Status']
    if all(col in df.columns for col in expected_columns):
        # Convert columns to appropriate data types
        for col in ['Week', 'Temp', 'Hum', 'Gas']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Prev_Status'] = label_encoder.transform(df[['Prev. Status']])
        df.dropna(subset=expected_columns, inplace=True)

        # Select features and scale them
        features = df[expected_columns]
        features_scaled = scaler.transform(features)

        # Predict using the model
        predictions = model.predict(features_scaled)
        df['Prediction'] = predictions

        # Add timestamp to the dataframe
        df['Timestamp'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])

        # Update session state plot data
        st.session_state.plot_data = pd.concat([st.session_state.plot_data, df[expected_columns + ['Prediction', 'Timestamp']]])

        # Plot results
        color_map = {'S': 'green', 'M': 'orange', 'U': 'red'}
        df['Color'] = df['Prediction'].map(color_map)
        fig, ax = plt.subplots()
        for label, color in color_map.items():
            subset = df[df['Prediction'] == label]
            ax.scatter(subset['Timestamp'], [label] * len(subset), color=color, label=label)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Prediction')
        ax.set_title('Real-Time Prediction Results')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        st.pyplot(fig)

        # Allow user to download the results
        csv = df.to_csv(index=False)
        st.download_button(label="Download Predictions", data=csv, file_name='predictions.csv', mime='text/csv')

if st.button('Fetch and Predict'):
    fetch_and_predict()

st.write("This section will refresh every 60 seconds to fetch new data and update predictions.")
while True:
    fetch_and_predict()
    time.sleep(60)

