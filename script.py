import streamlit as st
import pandas as pd
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import numpy as np
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
    
    # Check if 'Prev. Status' column exists, if not create it with a default value 'M'
    if 'Prev. Status' not in df.columns:
        st.warning("'Prev. Status' column is missing in the Google Sheet. Adding a default value 'M'.")
        df['Prev. Status'] = 'M'
    
    # Replace empty strings with 'M' in 'Prev. Status'
    df['Prev. Status'].replace('', 'M', inplace=True)
    
    # Convert columns to appropriate data types
    for col in ['Week', 'Temp', 'Hum', 'Gas']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle encoding of 'Prev. Status'
    try:
        df['Prev_Status'] = label_encoder.transform(df[['Prev. Status']])
    except ValueError as e:
        st.error(f"Error in transforming 'Prev. Status' column: {str(e)}")
        st.stop()

    # Drop rows with any NaN values
    df.dropna(subset=expected_columns, inplace=True)

    # Select features and scale them
    features = df[['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas']]
    features_scaled = scaler.transform(features)

    # Predict using the model
    predictions = model.predict(features_scaled)
    df['Prediction'] = predictions

    # Add timestamp to the dataframe
    if 'DATE' in df.columns and 'TIME' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
    else:
        df['Timestamp'] = pd.Timestamp.now()

    # Update session state plot data
    st.session_state.plot_data = pd.concat([st.session_state.plot_data, df[['Week', 'Temp', 'Hum', 'Gas', 'Prev_Status', 'Prediction', 'Timestamp']]])

    # Plot results using a stepwise plot with gradient background
    fig, ax = plt.subplots()
    
    # Fill the areas between levels with lighter colors
    ax.fill_between(df['Timestamp'], 0, 3, color='red', alpha=0.2)  # Red for U level
    ax.fill_between(df['Timestamp'], 3, 6, color='orange', alpha=0.2)  # Orange for M level
    ax.fill_between(df['Timestamp'], 6, 10, color='green', alpha=0.2)  # Green for S level
    
    # Use the midpoints of the timestamp for plotting
    df['Timestamp_numeric'] = df['Timestamp'].view(int)  # Convert timestamp to numeric for calculations
    midpoints = ((df['Timestamp_numeric'][:-1].values + df['Timestamp_numeric'][1:].values) / 2).astype('datetime64[ns]')

    # Step plot for the predictions
    ax.step(midpoints, df['Prediction'].map({'S': 10, 'M': 6, 'U': 3})[:-1], where='mid', color='black')
    
    ax.set_ylim(0, 11)
    ax.set_yticks([3, 6, 10])
    ax.set_yticklabels(['U', 'M', 'S'])

    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Prediction Status')
    ax.set_title('Real-Time Prediction Results')

    plt.xticks(rotation=45)
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








# def fetch_and_predict():
#     # Read data from the sheet
#     data = worksheet.get_all_values()
#     df = pd.DataFrame(data[1:], columns=data[0])

#     # Ensure required columns exist
#     expected_columns = ['Week', 'Temp', 'Hum', 'Gas', 'Prev. Status']
    
#     # Check if 'Prev. Status' column exists, if not create it with a default value 'M'
#     if 'Prev. Status' not in df.columns:
#         st.warning("'Prev. Status' column is missing in the Google Sheet. Adding a default value 'M'.")
#         df['Prev. Status'] = 'M'
#     else:
#         # Fill missing values in 'Prev. Status' with a default value 'M'
#         df['Prev. Status'].fillna('M', inplace=True)
    
#     # Convert columns to appropriate data types
#     for col in ['Week', 'Temp', 'Hum', 'Gas']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')

#     # Handle encoding of 'Prev. Status'
#     try:
#         # Fit the encoder on the known categories (only if it hasn't been fit before)
#         if 'Prev_Status' not in df.columns:
#             df['Prev_Status'] = label_encoder.fit_transform(df[['Prev. Status']])
#         else:
#             df['Prev_Status'] = label_encoder.transform(df[['Prev. Status']])
#     except ValueError as e:
#         st.error(f"Error in transforming 'Prev. Status' column: {str(e)}")
#         st.stop()

#     # Drop rows with any NaN values
#     df.dropna(subset=expected_columns, inplace=True)

#     # Select features and scale them
#     features = df[['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas']]
#     try:
#         features_scaled = scaler.transform(features)
#     except ValueError as e:
#         st.error(f"Error in scaling features: {str(e)}")
#         st.stop()

#     # Predict using the model
#     predictions = model.predict(features_scaled)
#     df['Prediction'] = predictions

#     # Add timestamp to the dataframe
#     if 'DATE' in df.columns and 'TIME' in df.columns:
#         df['Timestamp'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
#     else:
#         df['Timestamp'] = pd.Timestamp.now()

#     # Update session state plot data
#     st.session_state.plot_data = pd.concat([st.session_state.plot_data, df[['Week', 'Temp', 'Hum', 'Gas', 'Prev_Status', 'Prediction', 'Timestamp']]])

#     # Plot results
#     color_map = {'S': 'green', 'M': 'orange', 'U': 'red'}
#     df['Color'] = df['Prediction'].map(color_map)
#     fig, ax = plt.subplots()
#     for label, color in color_map.items():
#         subset = df[df['Prediction'] == label]
#         ax.scatter(subset['Timestamp'], [label] * len(subset), color=color, label=label)
#     ax.set_xlabel('Timestamp')
#     ax.set_ylabel('Prediction')
#     ax.set_title('Real-Time Prediction Results')
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     st.pyplot(fig)

#     # Allow user to download the results
#     csv = df.to_csv(index=False)
#     st.download_button(label="Download Predictions", data=csv, file_name='predictions.csv', mime='text/csv')

# if st.button('Fetch and Predict'):
#     fetch_and_predict()

