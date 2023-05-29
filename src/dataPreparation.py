"""Python file to split the data into train, test and cross validation data sets"""
import pandas as pd
from sklearn.model_selection import train_test_split
from dataPreprocess import loan_process_df

# Split the data into features (X) and Target (y)
X, y = loan_process_df.drop(columns="TARGET").values, loan_process_df["TARGET"].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=420
)

# Split test data further into test and cross validaiation sets
X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=0.5, random_state=420
)
