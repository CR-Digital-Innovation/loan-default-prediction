import streamlit as st
import pandas as pd
import requests

# Constants
API_URL = "https://saishivacr-cuddly-space-guide-xggj4x7r7xwf99r5-8000.preview.app.github.dev/predict"  # Update with your FAST API endpoint URL
MAX_FILE_SIZE = 10  # Maximum file size in MB

# Streamlit configuration
st.set_page_config(page_title="CSV Predictor", page_icon="🔮", layout="centered")


def main():
    # Header
    st.header("CSV Predictor")
    st.subheader("Upload a CSV file and get predictions")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        # Read CSV file
        df = pd.read_csv(uploaded_file)

        # Show CSV data as table
        st.subheader("CSV Data")
        st.write(df)

        # Predict button
        if st.button("Predict"):
            # Prepare data for API request
            payload = df.to_json()

            # Make API request to FAST API
            response = requests.post(API_URL, json=payload)

            # Process API response
            if response.status_code == 200:
                results = response.json()

                # Display prediction results
                st.subheader("Prediction Results")
                st.table(results)
            else:
                st.error("Failed to retrieve prediction results. Please try again.")

    # Footer
    st.write("---")
    st.text("Customizable footer note")


if __name__ == "__main__":
    main()
