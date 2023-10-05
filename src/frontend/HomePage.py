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

    #custom colors
    colors = {
        "Confirmed Repayer": "#a3b966",
        "Probable Defaulter": "#eb9b56",
        "Confirmed Defaulter": "#b92e36",
    }

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
                input_data = pd.DataFrame(csv_data)

                # columns required from the input data
                columns_to_select_from_input_df = ['SK_ID_CURR','NAME_EDUCATION_TYPE', 'REGION_RATING_CLIENT', 'AMT_INCOME_TOTAL','NAME_INCOME_TYPE','DAYS_EMPLOYED']
                df_charts = pd.merge(predict_result_df, input_data[columns_to_select_from_input_df], on='SK_ID_CURR', how='inner')

            

                # Creating bins for Employement Time
                df_charts['YEARS_EMPLOYED'] = df_charts['DAYS_EMPLOYED'].abs() // 365
                bins = [0,5,10,20,30,40,50,60,150]
                slots = ['0-5','5-10','10-20','20-30','30-40','40-50','50-60','60 above']

                # Creating bins for income total
                bin_edges = [0, 100000, 300000, 600000, 900000, float('inf')]
                bin_labels = ['Less than 1L', '1L-3L', '3L-6L', '6L-9L', 'More than 9L']

                # Create a new column 'Income Range' based on the bins
                df_charts['Income Range'] = pd.cut(df_charts['AMT_INCOME_TOTAL'], bins=bin_edges, labels=bin_labels, right=False)


                df_charts['EMPLOYMENT_YEAR']=pd.cut(df_charts['YEARS_EMPLOYED'],bins=bins,labels=slots)
            
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

            plot_col1, plot_col2 = st.columns(2)

            # distribution plot for loan risk
            prediction_dist = df_charts['Prediction'].value_counts()

            fig = px.pie(prediction_dist, values=prediction_dist.values, names=prediction_dist.index, title='Prediction Distribution',color=prediction_dist.index,)

            # Apply custom colors based on the 'colors' dictionary
            for i, sector in enumerate(fig.data[0].labels):

                # Convert the tuple to a list to modify it
                marker_colors = list(fig.data[0].marker.colors)
                marker_colors[i] = colors.get(sector, "#1f77b4")
                fig.data[0].marker.colors = tuple(marker_colors)  
  

            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=.5),paper_bgcolor='#F1F1F1',plot_bgcolor = '#F1F1F1',title_x = 0.4)
            plot_col1.plotly_chart(fig, use_container_width=True)

            
            # plot the graph for income type #

            # Group data by 'NAME_INCOME_TYPE' and 'Prediction' and calculate the percentage within each 'NAME_INCOME_TYPE' category
            grouped_data = df_charts.groupby(['NAME_INCOME_TYPE', 'Prediction']).size().reset_index(name='Count')

            # Create a grouped bar chart with data labels as percentages
            fig = px.bar(grouped_data, x='NAME_INCOME_TYPE', y='Count', color='Prediction',
                        labels={'Prediction': 'Prediction','Count':'Count'},
                        text='Count', title='Prediction Trends Over Income Type')

            # Apply custom colors based on the 'colors' dictionary
            for i, category in enumerate(fig.data):
                if category.name in colors:
                    fig.data[i].marker.color = colors[category.name]

            fig.update_traces(textangle=0, textposition='inside', insidetextanchor='middle')
            fig.update_xaxes(title='Type of Income')
            fig.update_yaxes(title='Count')
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=.5,title=""),paper_bgcolor='#F1F1F1',plot_bgcolor = '#F1F1F1',title_x = 0.3)

            # Display the chart in Streamlit
            plot_col2.plotly_chart(fig,use_container_width=True)


            # plot the graph for employement years #

            # Group data by 'Prediction' and 'NAME_INCOME_TYPE' and calculate the percentage
            grouped_data = df_charts.groupby(['EMPLOYMENT_YEAR', 'Prediction']).size().reset_index(name='Count')
            
            # Create a grouped bar chart with data labels as percentages
            fig = px.bar(grouped_data, x='EMPLOYMENT_YEAR', y='Count', color='Prediction',
                        labels={'Prediction': 'Prediction', 'Count': 'Count'},
                        text='Count', title='Prediction Breakdown by Employment Years')

            # Apply custom colors based on the 'colors' dictionary
            for i, category in enumerate(fig.data):
                if category.name in colors:
                    fig.data[i].marker.color = colors[category.name]

            fig.update_traces(textangle=0, textposition='inside', insidetextanchor='middle')
            fig.update_xaxes(title='Employment in years')
            fig.update_yaxes(title='Count')
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=.5,title=""),paper_bgcolor='#F1F1F1',plot_bgcolor = '#F1F1F1',title_x = 0.3)

            # Display the chart in Streamlit
            plot_col1.plotly_chart(fig,use_container_width=True)

            # plot for income range for the applicant


            # Create a histogram with custom bins
            grouped_data = df_charts.groupby(['Income Range', 'Prediction']).size().reset_index(name='Count')

            # Create a grouped bar chart with data labels as percentages
            fig = px.bar(grouped_data, x='Income Range', y='Count', color='Prediction',
                        labels={'Prediction': 'Prediction', 'Count': 'Count'},
                        text='Count', title='Prediction Trends Over Income Range')

            # Apply custom colors based on the 'colors' dictionary
            for i, category in enumerate(fig.data):
                if category.name in colors:
                    fig.data[i].marker.color = colors[category.name]

            fig.update_traces(textangle=0, textposition='inside', insidetextanchor='middle')
            fig.update_xaxes(title='Total Income Values')
            fig.update_yaxes(title='Count')
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=.5,title=""),paper_bgcolor='#F1F1F1',plot_bgcolor = '#F1F1F1',title_x = 0.35)

            # Display the chart in Streamlit
            plot_col2.plotly_chart(fig,use_container_width=True)

        else:
            st.error("Please upload appropriate CSV file or Paste appropriate CSV data from provided samples.")


if __name__ == "__main__":
    main()
