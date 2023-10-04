"""Streamlit frontend to accept CSV files or CSV data and """
# import sys
import os
import csv
import requests
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go

# Get the absolute path to the project root directory
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root directory to the PYTHONPATH
# sys.path.insert(0, project_root)

# Importing required modules from utils
# from utils.load_EnvVars import BACKEND_HOST_URL

# Loading Backend Host URL from Env variable
# BACKEND_HOST_URL = "http://127.0.0.1:8000"
BACKEND_HOST_URL = os.getenv("BACKEND_HOST_URL")

# Setting colors for the prediction results

colors = {
    "Confirmed Repayer": "#a3b966",
    "Probable Defaulter": "#eb9b56",
    "Confirmed Defaulter": "#b92e36",
}


# Streamlit configuration
st.set_page_config(
    page_title="Loan-default-risk-predictor",
    page_icon="⚙️",
    layout="wide",
    menu_items={
        "About": "A simple interface to upload the application data as a csv file or paste in csv format and get predictions whether the applicant is a repayer or a defaulter."
    },
)
st.markdown(
    """ <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .footer {
        position: fixed;
        left: 80px;
        bottom: 0;
        width: 100%;
        color: white;
    }
    </style> 
    <div class="footer">
    <p>©️ 2023, CriticalRiver Inc. All rights reserved.</p>
    </div> """,
    unsafe_allow_html=True,
)


def process_csv(file_contents):
    csv_input = file_contents.decode("utf-8").splitlines()
    csv_reader = csv.reader(csv_input)
    csv_data = []
    header = next(csv_reader)  # Extract the header row
    for row in csv_reader:
        if not all(cell.strip() == "" for cell in row):  # Skip empty rows
            row_dict = {}
            for i, value in enumerate(row):
                if value.strip() == "":
                    row_dict[header[i]] = None  # Assign None for missing values
                else:
                    try:
                        row_dict[header[i]] = int(value)  # Convert to integer
                    except ValueError:
                        try:
                            row_dict[header[i]] = float(value)  # Convert to float
                        except ValueError:
                            row_dict[
                                header[i]
                            ] = value  # Store as string if conversion fails
            csv_data.append(row_dict)
    return csv_data


def upload_file():
    upload_file = st.file_uploader("Upload CSV file", type=["csv"])
    if upload_file is not None:
        content = upload_file.read()
        csv_data = process_csv(content)
        return csv_data


def text_input():
    text_input = st.text_area(
        "Paste CSV text",
        help="Type or paste csv text data and press 'ctrl'+'Enter' to apply",
        placeholder="ALPHABET, NUMBER\nA, 1\nB,2",
    )
    if text_input:
        csv_data = process_csv(text_input.encode())
        return csv_data


def predict_df(df: pd.DataFrame):
    # Prepare dataframe for input to the API Endpoint
    if not df.columns.empty:
        json_data = df.to_json(orient="records")
        headers = {"Content-Type": "application/json"}
        json_payload = {"id_process": "0", "dataframe": json.loads(json_data)}

        # Prepare the API request
        API_ENDPOINT = f"{BACKEND_HOST_URL}/predict"  # To parse dataframe

        # Get response
        response = requests.post(
            API_ENDPOINT, json=json_payload, headers=headers
        )  # To parse dataframe
        if response.status_code == 200:
            predict_results = response.json()

            predict_result_df = pd.DataFrame.from_dict(predict_results)
            # st.write(predict_results)

            return predict_result_df


def highlight_prediction(value):
    color = colors[value]

    return f"background-color: {color}"


def predict_csv(csv_data: list):
    # Prepare the API request
    API_ENDPOINT = f"{BACKEND_HOST_URL}/csv_predict"  # To parse CSV data

    # Get response
    response = requests.post(API_ENDPOINT, json=csv_data)  # To parse CSV data
    if response.status_code == 200:
        return response.json()


def main():
    # Header
    st.image(
        "https://www.criticalriver.com/wp-content/uploads/2022/04/cr-logo-updated.png"
    )
    st.header("Loan Default Risk Predictor")
    st.subheader("Upload the application data in csv format and get predictions")

    st.write("Select an option to either upload a csv file or paste data in csv format")
    st.markdown("Download demo files from [here](https://github.com/CR-Digital-Innovation/loan-default-prediction/tree/develop/data)")

    option = st.selectbox("Choose an option", ("Upload a CSV File", "Paste CSV Text"))

    if option == "Upload a CSV File":
        csv_data = upload_file()
        input_df = pd.DataFrame(csv_data)
        st.write("**Extracted Information:**")
        st.write(input_df)

    elif option == "Paste CSV Text":
        csv_data = text_input()
        input_df = pd.DataFrame(csv_data)
        st.write("**Extracted Information:**")
        st.write(input_df)

    if st.button("Predict"):
        if csv_data is not None:
            with st.spinner("Predicting loan default risk..."):
                predict_results = predict_csv(csv_data)  # Predict with CSV data
                # predict_result_df = predict_df(input_df)  # Predict with dataframe

                predict_result_df = pd.read_json(predict_results["predictions"])

                # Apply color style to specific column
                predict_results_styled_df = predict_result_df.style.applymap(
                    highlight_prediction, subset=["Prediction"]
                )
                accuracy = round(predict_results["accuracy"] * 100, 2)

            col1, col2, col3 = st.columns([0.4,0.2,0.4])
            

            with col1:
                st.write("**Predictions:**\n", predict_results_styled_df)

            with col2:
                st.empty()
                st.metric(label="**Accuracy**", value=f"{accuracy}%", help="Accuracy is a measure of the model's ability to correctly predict or classify data points.")
                st.metric(label="**F1 Score**", value=round(predict_results["f1_score"], 2), help="F1 Score is a single metric that combines precision and recall to assess a model's overall performance in binary classification tasks. A higher F1 score indicates a model with better overall accuracy and effectiveness in binary classification tasks.")
            with col3:
                cm = pd.DataFrame(predict_results["cm"])
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted Labels", y="Actual Labels"),
                    x=["Confirmed Repayers", "Confirmed Defaulters"],
                    y=["Confirmed Repayers", "Confirmed Defaulters"],
                    title="Confusion Matrix",
                    height=400,
                )
                st.plotly_chart(
                    fig_cm, use_container_width=True, config={"displayModeBar": False}
                )
                # st.write(cm)

            # Create a histogram chart using Plotly
            fig = go.Figure()

            # Iterate over unique prediction results and add the bars
            for prediction in predict_result_df["Prediction"].unique():
                subset = predict_result_df[predict_result_df["Prediction"] == prediction]
                fig.add_trace(
                    go.Bar(
                        x=[str(id_) for id_ in subset["SK_ID_CURR"]],
                        y=subset["Confidence Score in %"],
                        name=prediction,
                        marker_color=colors[prediction],
                        # width=1
                    )
                )

            # Set chart title and axes labels
            fig.update_layout(
                title="Prediction results",
                xaxis_title="SK_ID_CURR",
                yaxis_title="Confidence Score (%)",
                xaxis=dict(
                    type="category",
                    categoryorder="array",
                    categoryarray=[str(id_) for id_ in predict_result_df["SK_ID_CURR"]],
                    tickangle=-90,
                ),
            )

            # Display the histogram chart
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.error("Please upload appropriate CSV file or Paste appropriate CSV data from provided samples.")


if __name__ == "__main__":
    main()
