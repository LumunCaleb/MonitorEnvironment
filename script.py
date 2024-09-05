import streamlit as st
import pandas as pd
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import numpy as np
import time

# # Define the paths to your joblib files
# Define the paths to your joblib files
joblib_file_path = 'knnmainnew_model.joblib'
label_encoder_path = 'encoder.joblib'
scaler_path = 'scaler.joblib'

# Load the model, label encoder, and scaler using joblib
model = joblib.load(joblib_file_path)
label_encoder = joblib.load(label_encoder_path)
scaler = joblib.load(scaler_path)


# Google Sheets API setup
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["google_api"], scope)
client = gspread.authorize(creds)
sheet_url = 'https://docs.google.com/spreadsheets/d/1lbGCOmPlX4HXzNW2WDfocolRO6E28uFGTNeeH_yBIbo/edit#gid=0'
sheet = client.open_by_url(sheet_url)
worksheet = sheet.get_worksheet(0)

st.title("Environmental Monitoring App")

# # Initialize session state variables
if 'previous_prediction' not in st.session_state:
    st.session_state.previous_prediction = 'M'
if 'plot_data' not in st.session_state:
    st.session_state.plot_data = pd.DataFrame(columns=['Week', 'Temp', 'Hum', 'Gas', 'Prev_Status', 'Prediction', 'Timestamp'])

# Initialize session state variables to track previous prediction
if 'previous_prediction' not in st.session_state:
    st.session_state.previous_prediction = 'M'  # Default value or set to 'unknown'

#Custom CSS to set the background color, text color, and add a watermark
st.markdown(
    """
    <style>
    .main {
        background-color: #004d00;  /* Dark green background */
        color: white;  /* White text color */
    }
    .css-18e3th9 { /* Header */
        background-color: #003300;  /* Darker green for header */
        color: Black;  /* White text color */
    }
    .css-1j7a9z2 { /* Button */
        background-color: #006400;  /* Dark green for button */
        color: white;  /* White text color on button */
    }
    .css-1n7v3w8 input { /* Input fields */
        color: white;  /* White text color in input fields */
        background-color: gray;  /* Gray background for input fields */
    }
    .css-1v3h5q2 { /* Text Input */
        color: white;  /* White text color in text input fields */
        background-color: gray;  /* Gray background for text input fields */
    }
    .css-10trblm { /* Title */
        color: gray;  /* White text color for the title */
    }
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 10px;
        opacity: 0.5;
        z-index: -1;
        pointer-events: none;
    }
    </style>
    <div class="watermark">
        <img src="https://example.com/path-to-your-watermark-image.png" alt="Poultry Farm" width="200"/>
    </div>
    """,
    unsafe_allow_html=True
)


# Sidebar for navigation
option = st.sidebar.selectbox(
    "Choose an option",
    ("Predict with User Input", "Upload a CSV File", "View Real Time Prediction")
)

if option == "Predict with User Input":
    st.write("Enter feature values for prediction:")

    # Input fields
    Week = st.number_input('Week', value=0, step=1)  # Use step=1 for integer input
    Previous_Status = st.text_input('Previous_Status', st.session_state.previous_prediction)
    Temperature = st.number_input('Temperature', value=0.0)
    Humidity = st.number_input('Humidity', value=0.0)
    GasLevel = st.number_input('GasLevel', value=0.0)

    # Create a DataFrame for input features
    cols = ['Week', 'Prev. Status', 'Temp', 'Hum', 'Gas']
    input_data = pd.DataFrame([[Week, Previous_Status, Temperature, Humidity, GasLevel]], columns=cols)

    # Print the DataFrame to debug
    st.write("Input DataFrame for Prediction:")
    st.write(input_data)

    # Transform 'Prev. Status' using the label encoder
    try:
        # Ensure the 'Prev. Status' column is treated as a 2D array with one feature
        prev_status_encoded = label_encoder.transform(input_data[['Prev. Status']].values.reshape(-1, 1))
        input_data['Prev_Status'] = prev_status_encoded
    except ValueError as e:
        # Handle categories not in encoder
        st.error(f"Encoding error: {e}")
        input_data['Prev_Status'] = label_encoder.transform([['unknown']])[0]

    # Select the required features and scale them
    input_data = input_data[['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas']]
    input_data_scaled = scaler.transform(input_data)

    if st.button('Predict'):
        prediction = model.predict(input_data_scaled)[0]
        st.write(f'Prediction: {prediction}')

        # Update session state with the latest prediction
        st.session_state.previous_prediction = prediction

        # Update previous predictions DataFrame
        new_prediction = pd.DataFrame({
            'Week': [Week],
            'Temperature': [Temperature],
            'Humidity': [Humidity],
            'GasLevel': [GasLevel],
            'Predicted Status': [prediction]
        })
        
        st.session_state.previous_predictions = pd.concat([st.session_state.previous_predictions, new_prediction], ignore_index=True)

        # Display previous predictions and the legend side by side
        col1, col2 = st.columns([4, 2])

        with col1:
            st.write("**Previous Predictions**")
            st.dataframe(st.session_state.previous_predictions)

        with col2:
            st.write("**Prediction Legend**")
            st.write("**U**: Unsafe", "**M**: Moderately Safe", "**S**: Safe")

elif option == "Upload a CSV File":
    st.write("Upload a CSV file to update:")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(uploaded_file)

            # Display the first few rows of the DataFrame to confirm it's loaded correctly
            st.write("CSV file loaded successfully. Here are the first few rows:")
            st.write(df.head())

            # Add a button to trigger the prediction
            if st.button('Predict from CSV'):
                # Check if 'Prev. Status' column exists
                if 'Prev. Status' not in df.columns:
                    st.warning("'Prev. Status' column is missing. It will be created.")
                    df['Prev. Status'] = 'M'  # Initialize with a default value

                # Handle NaN values in 'Prev. Status'
                df['Prev. Status'].fillna('M', inplace=True)

                # Define the expected columns and their types
                expected_columns = {
                    'Week': 'int',
                    'Prev. Status': 'object',
                    'Temp': 'float',
                    'Hum': 'float',
                    'Gas': 'float'
                }

                # Validate columns
                missing_columns = [col for col in expected_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"Missing columns: {', '.join(missing_columns)}")
                else:
                    # Validate and correct data types
                    for col, dtype in expected_columns.items():
                        if col in df.columns:
                            if dtype == 'float':
                                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                            elif dtype == 'int':
                                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')  # 'Int64' to allow for missing values
                            elif dtype == 'object':
                                df[col] = df[col].astype(str)

                    # Check for any remaining issues
                    if df.isnull().values.any():
                        st.error("Data contains null values. Please clean the data.")
                        st.write(df[df.isnull().any(axis=1)])
                    else:
                        # Ensure 'Prev. Status' is correctly encoded
                        prev_status_unique = df['Prev. Status'].unique()
                        categories = label_encoder.categories_[0]
                        missing_categories = [x for x in prev_status_unique if x not in categories]
                        
                        if missing_categories:
                            st.warning(f"Some categories in 'Prev. Status' are not in the encoder: {missing_categories}")
                            # Replace missing categories with 'unknown'
                            df['Prev. Status'] = df['Prev. Status'].apply(lambda x: 'unknown' if x not in categories else x)

                        # Transform 'Prev. Status' using the label encoder
                        try:
                            df['Prev_Status'] = label_encoder.transform(df[['Prev. Status']].values.reshape(-1, 1))
                        except ValueError as e:
                            st.error(f"Encoding error: {e}")
                            df['Prev_Status'] = label_encoder.transform([['unknown']])[0]

                        # Process DataFrame
                        features = df[['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas']]
                        features_scaled = scaler.transform(features)
                        df['Prediction'] = model.predict(features_scaled)

                        # Update 'Prev. Status' with the latest prediction
                        prev_prediction = 'U'
                        for i in range(len(df)):
                            df.at[i, 'Prev. Status'] = prev_prediction
                            prev_prediction = df.at[i, 'Prediction']

                        # Map prediction results to colors
                        color_map = {'S': 'green', 'M': 'orange', 'U': 'red'}
                        df['Color'] = df['Prediction'].map(color_map)
                        # Plot results
                        fig, ax = plt.subplots()
                        for label, color in color_map.items():
                            subset = df[df['Prediction'] == label]
                            ax.scatter(subset.index, [label] * len(subset), color=color, label=label)

                        ax.set_xlabel('Index')
                        ax.set_ylabel('Prediction')
                        ax.set_title('Prediction Results')
                        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                        st.pyplot(fig)

                        # Prepare DataFrame to download
                        download_df = pd.DataFrame({
                            'Temperature': [25, 30, 35],
                            'Humidity': [60, 65, 70],
                            'Gas Level': [300, 320, 340],
                            'Weeks of Poultry Birds': [2, 3, 4],
                            'Predicted Status': ['M', 'U', 'S']
                        })

                        # Convert DataFrame to CSV
                        csv = download_df.to_csv(index=False)
                        
                        # Create a download button
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name='predicted_results.csv',
                            mime='text/csv',
                            key='download-csv'
                        )

        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")

elif option == "View Real Time Prediction":
    st.write("View Real Time Prediction")
    # st.write("Real-Time Prediction from Google Sheets:")
    if 'previous_predictions' not in st.session_state:
        st.session_state.previous_predictions = pd.DataFrame(columns=['Week', 'Temperature', 'Humidity', 'GasLevel', 'Predicted Status'])

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
        ax.fill_between(df['Timestamp'], 0, 3, color='red', alpha=1)  # Red for U level
        ax.fill_between(df['Timestamp'], 3, 6, color='orange', alpha=1)  # Orange for M level
        ax.fill_between(df['Timestamp'], 6, 10, color='green', alpha=1)  # Green for S level
        
        # Use the midpoints of the timestamp for plotting
        df['Timestamp_numeric'] = df['Timestamp'].view(int)  # Convert timestamp to numeric for calculations
        midpoints = ((df['Timestamp_numeric'][:-1].values + df['Timestamp_numeric'][1:].values) / 2).astype('datetime64[ns]')
    
        # Step plot for the predictions
        ax.step(midpoints, df['Prediction'].map({'S': 8, 'M': 4.5, 'U': 1.5})[:-1], where='mid', color='black')
        
        ax.set_ylim(0, 10)
        ax.set_yticks([1.5, 4.56, 8])
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
        ax.fill_between(df['Timestamp'], 0, 3, color='red', alpha=1)  # Red for U level
        ax.fill_between(df['Timestamp'], 3, 6, color='orange', alpha=1)  # Orange for M level
        ax.fill_between(df['Timestamp'], 6, 10, color='green', alpha=1)  # Green for S level
        
        # Use the midpoints of the timestamp for plotting
        df['Timestamp_numeric'] = df['Timestamp'].view(int)  # Convert timestamp to numeric for calculations
        midpoints = ((df['Timestamp_numeric'][:-1].values + df['Timestamp_numeric'][1:].values) / 2).astype('datetime64[ns]')
    
        # Step plot for the predictions
        ax.step(midpoints, df['Prediction'].map({'S': 8.3, 'M': 4.95, 'U': 1.65})[:-1], where='mid', color='black')
        
        ax.set_ylim(0, 10)
        ax.set_yticks([1.65, 4.95, 8.3])
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
