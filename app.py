import streamlit as st
import pandas as pd
import requests

# Streamlit App
st.title("CSV File Predictor")
st.write("Upload a CSV file to get predictions")

# Upload CSV File
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    # Read CSV File
    df = pd.read_csv(uploaded_file)

    # Display CSV Data
    st.write("Uploaded CSV Data:")
    st.dataframe(df)

    # Send CSV Data to FastAPI for Prediction
    if st.button("Predict"):
        # Prepare Data for Request
        csv_data = df.to_csv(index=False)
        files = {"file": ("data.csv", csv_data)}

        # Make POST Request to FastAPI
        response = requests.post("http://localhost:8000/predict", files=files)

        if response.status_code == 200:
            # Parse Response Data
            result = response.json()

            # Display Predictions
            st.write("Predictions:")
            predictions = result["predictions"]
            probabilities = result["probabilities"]
            prediction_df = pd.DataFrame(
                {
                    "CSV Column": df.iloc[:, 0],  # First column of the CSV
                    "Prediction": predictions,
                    "Probability (Class 0)": [p[0] for p in probabilities],
                    "Probability (Class 1)": [p[1] for p in probabilities],
                }
            )
            st.dataframe(prediction_df)
        else:
            st.write("Error occurred during prediction.")
