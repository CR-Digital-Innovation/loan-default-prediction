
import sys
import os
import csv
import requests
import streamlit as st
import pandas as pd
import json

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root directory to the PYTHONPATH
sys.path.insert(0, project_root)

# Importing required modules from utils
from utils.load_EnvVars import BACKEND_HOST_URL

# Streamlit configuration
st.set_page_config(page_title="Loan-default-risk-predictor", page_icon="⚙️", layout="wide",  menu_items={'About': "A simple interface to upload the application data as a csv file or paste in csv format and get predictions whether the applicant is a repayer or a defaulter."})
st.markdown(""" <style>
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> """, unsafe_allow_html=True)


def process_csv(file_contents):
    csv_input = file_contents.decode('utf-8').splitlines()
    csv_reader = csv.reader(csv_input)
    csv_data = []
    header = next(csv_reader)  # Extract the header row
    for row in csv_reader:
        if not all(cell.strip() == '' for cell in row):  # Skip empty rows
            row_dict = {}
            for i, value in enumerate(row):
                if value.strip() == '':
                    row_dict[header[i]] = None  # Assign None for missing values
                else:
                    try:
                        row_dict[header[i]] = int(value)  # Convert to integer
                    except ValueError:
                        try:
                            row_dict[header[i]] = float(value)  # Convert to float
                        except ValueError:
                                row_dict[header[i]] = value  # Store as string if conversion fails
            csv_data.append(row_dict)
    return csv_data


def upload_file():
    upload_file = st.file_uploader("Upload CSV file", type=["csv"])
    if upload_file is not None:
        content = upload_file.read()
        csv_data = process_csv(content)
        return csv_data
    

def text_input():
    text_input = st.text_area("Paste CSV text", help="Type or paste csv text data and press 'ctrl'+'Enter' to apply", placeholder="ALPHABET, NUMBER\nA, 1\nB,2")
    if text_input:
        csv_data = process_csv(text_input.encode())
        return csv_data


def predict_df(df: pd.DataFrame):
    # Prepare dataframe for input to the API Endpoint
    if not df.columns.empty:
        json_data = df.to_json(orient='records')
        headers = {'Content-Type': 'application/json'}
        json_payload = {
            "id_process": '0',
            "dataframe": json.loads(json_data)
            }
        
        # Prepare the API request
        API_ENDPOINT = f"{BACKEND_HOST_URL}/predict"  # To parse dataframe

        # Get response
        response = requests.post(API_ENDPOINT, json=json_payload, headers=headers)# To parse dataframe
        if response.status_code == 200:
            predict_results = response.json()

            predict_result_df = pd.DataFrame.from_dict(predict_results)
            #st.write(predict_results)

            return predict_result_df


def get_prediction(row):
    # Function to deterimine prediction category and confidence score
    class_0_prob = row['Probability 0']
    class_1_prob = row['Probability 1']

    if class_1_prob < 0.4:
        prediction = 'Repayer'
        confidence_score = class_0_prob * 100
    elif class_1_prob < 0.7:
        prediction = 'Repayer with Risk'
        confidence_score = max(class_0_prob, class_1_prob) * 100
    else:
        prediction = 'Defaulter'
        confidence_score = class_1_prob * 100
    
    return prediction, confidence_score


def highlight_prediction(value):

    if value == 'Defaulter':
        color = '#b92e36'
    elif value == 'Repayer':
        color = '#a3b966'
    else:
        color = '#eb9b56'

    return f'background-color: {color}'


def predict_csv(csv_data: list):
    # Prepare the API request
    API_ENDPOINT = f"{BACKEND_HOST_URL}/csv_predict"   # To parse CSV data

    # Get response
    response = requests.post(API_ENDPOINT, json=csv_data)   # To parse CSV data
    if response.status_code == 200:
        predict_results = response.json()

        predict_result_df = pd.DataFrame.from_dict(predict_results)
        #st.write(predict_results)

        # Apply transformation to create the new DataFrame
        predict_result_df['Prediction'], predict_result_df['Confidence Score'] = zip(*predict_result_df.apply(get_prediction, axis=1))

        # Select and reorder the desired columns
        predict_desc_df = predict_result_df[['SK_ID_CURR', 'Prediction', 'Confidence Score']]

        # Apply color style to specific column
        predict_desc_styled_df = predict_desc_df.style.applymap(highlight_prediction, subset=['Prediction'])

        return predict_desc_styled_df
        

def main():
    # Header
    st.header("Loan Default Predictor")
    st.subheader("Upload a application data in csv format and get predictions")

    st.write("Select an option to either upload a csv file or paste data in csv format")

    option = st.selectbox("Choose an option", ("Upload a CSV File", "Paste CSV Text"))

    if option == "Upload a CSV File":
        csv_data = upload_file()
        input_df = pd.DataFrame(csv_data)
        st.write("Extracted Information:")
        st.write(input_df)
            
    elif option == "Paste CSV Text":
            csv_data = text_input()
            input_df = pd.DataFrame(csv_data)
            st.write("Extracted Information:")
            st.write(input_df)
        
    if st.button("Predict"):
        predict_result_df = predict_csv(csv_data)  # Predict with CSV data
        # predict_result_df = predict_df(input_df)  # Predict with dataframe
        st.write(predict_result_df)
                 

if __name__ == "__main__":
    main()