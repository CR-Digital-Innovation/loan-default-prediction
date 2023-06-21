import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

def main():

    # Sample DataFrame
    data = {
        'ID': [1, 2, 3, 4, 5],
        'Predict Class': [0, 1, 1, 0, 1],
        'Class 0 Probability': [0.75, 0.6, 0.3, 0.9, 0.2],
        'Class 1 Probability': [0.25, 0.4, 0.7, 0.1, 0.8],
        'Actual Target': [0, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data)

    # Function to determine prediction category and confidence score
    def get_prediction(row):
        class_0_prob = row['Class 0 Probability']
        class_1_prob = row['Class 1 Probability']
        
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
    
    # Function to calculate accuracy
    def calculate_accuracy(row):
        if row['Predict Class'] == row['Actual Target']:
            return 1
        else:
            return 0

    # Calculate accuracy for each row
    df['Accuracy'] = df.apply(calculate_accuracy, axis=1)

    # Calculate cumulative accuracy
    df['Cumulative Accuracy'] = df['Accuracy'].cumsum() / df.index


    # Apply transformation to create the new DataFrame
    df['Prediction'], df['Confidence Score'] = zip(*df.apply(get_prediction, axis=1))

    # Select and reorder the desired columns
    new_df = df[['ID', 'Prediction', 'Confidence Score']]

    # Display the new DataFrame
    # print(new_df)

    # Function to apply color style to DataFrame cells
    # Function to apply color style to specific column
    def highlight_column2(value):
        color = '#b92e36' if value else '#a3b966'
        return f'background-color: {color}'

    def highlight_prediction(value):

        if value == 'Defaulter':
            color = '#b92e36'
        elif value == 'Repayer':
            color = '#a3b966'
        else:
            color = '#eb9b56'

        return f'background-color: {color}'

    # Apply color style to specific column
    styled_df = df.style.applymap(highlight_column2, subset=['Predict Class'])

    # Display styled DataFrame using Streamlit
    st.write(styled_df)
    #st.table(styled_df)

    st.write(new_df.style.applymap(highlight_prediction, subset=['Prediction']))

    # Convert 'ID' column to string datatype
    df['ID'] = df['ID'].astype(str)

    # Set the style of seaborn
    sns.set_style("darkgrid")
    
    # Create line plots for 'Actual' and 'Predicted'
    plt.figure(figsize=(10, 6))  # Adjust the size of the figure

    sns.lineplot(data=df, x='ID', y='Actual Target', label='Actual', color='blue')
    sns.lineplot(data=df, x='ID', y='Predict Class', label='Predicted', color='red')

    # Set chart title and axes labels
    plt.title('Actual vs Predicted')
    plt.xlabel('ID')
    plt.ylabel('Target')

    # Add legend
    plt.legend()

    # Display the line chart
    st.pyplot()

    # Plot line chart for accuracy
    st.line_chart(df['Cumulative Accuracy'])


if __name__ == "__main__":
    main()