"""Function to load and pre process the data for model training"""
import pandas as pd
from globalVars import (
    AWS_S3_BUCKET,
    AWS_S3_DATA_DIRECTORY,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


def load_dataframe(fileName: str) -> pd.DataFrame:
    """Loading csv file that is present in AWS S3 Bucket as dataframe using Pandas"""

    dataframe = pd.read_csv(
        f"s3://{AWS_S3_BUCKET}/{AWS_S3_DATA_DIRECTORY}/{fileName}",
        storage_options={
            "key": AWS_ACCESS_KEY_ID,
            "secret": AWS_SECRET_ACCESS_KEY,
        },
    )
    return dataframe


def null_value_df(df: pd.DataFrame) -> pd.DataFrame:
    """This function takes the dataframe as input and provides another dataframe with \
        column names and percentage of null values in it"""

    null_values = round(df.isnull().sum() / df.shape[0] * 100.00, 2)
    null_df = pd.DataFrame(null_values.reset_index())
    null_df.columns = ["Column Name", "Null values percentage"]
    return null_df


def null_value_column_list(df: pd.DataFrame, percentage: float) -> list:
    """This function takes the dataframe, and float percentage value and provides \
        the list of column names with greater than or equal to the defined percentage"""

    null_df = null_value_df(df)
    column_list = null_df[null_df["Null values percentage"] >= percentage][
        "Column Name"
    ].tolist()
    return column_list


def convert_obj_to_cat(df: pd.DataFrame) -> pd.DataFrame:
    """This function takes the dataframe and converts the object datatype columns into categorical columns"""

    obj_columns = df.select_dtypes(include="object").columns.tolist()
    for col in obj_columns:
        df[col] = pd.Categorical(df[col])
    return df


def handle_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """This function takes the dataframe as input and imputes the null values in following ways:
    A. For numerical category type columns:
        1. If null value percentage is less than or equal to 15% then median() is used.
        2. If null value is above 15% then mean() is used.
    B. For Categorical type columns:
        1. If null value percentage is less than or equal to 15% then mode() is applied.
        2. If null value is above 15% then new "Unknown" category is created."""

    null_percentages = df.isnull().sum() / df.shape[0] * 100.00

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            # Numerical type column
            null_percentage = null_percentages[column]
            if null_percentage <= 15:
                # Apply median if null percentage is less than or equal to 15%
                df[column].fillna(df[column].median(), inplace=True)
            else:
                # Apply mean if null percentage is more than 15%
                df[column].fillna(df[column].mean(), inplace=True)
        elif pd.api.types.is_categorical_dtype(df[column]):
            # Categorical type column
            null_percentage = null_percentages[column]
            if null_percentage <= 15:
                # Apply mode if null percentage is less than or equal to 15%
                df[column].fillna((df[column].mode()[0]), inplace=True)
            else:
                # Create an unknown category if null percentage is more than 15%
                df[column] = df[column].cat.add_categories("Unknown")
                df[column].fillna("Unknown", inplace=True)
    return df


def label_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """This function converts the categorical columns into numerical columns using labelencoder"""

    for column in df.columns:
        if pd.api.types.is_categorical_dtype(df[column]):
            df[column] = label_encoder.fit_transform(df[column])
    return df

