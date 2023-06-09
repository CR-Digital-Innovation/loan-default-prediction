"""Few useful data preprocessing functions"""

import pandas as pd
from typing import Tuple
from categorical_Encoder import CategoricalEncoder # Custom LabelEncoder() technique through ColumnTransfer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer


def null_value_df(df: pd.DataFrame) -> pd.DataFrame:
    """This function takes the dataframe as input and provides another dataframe with \
        column names and percentage of null values in it"""

    null_values = round(df.isnull().mean() * 100.00, 2)
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


def custom_standardization(df: pd.DataFrame) -> pd.DataFrame:
    """ This function to handle the custom data standardization techniques to handle the oulier scenarios with few columns.
        This process must be done for the clean data as well inorder to make the valid predictions.
    """
    # Standardizing DAYS_EMPLOYED values
    df.loc[df["DAYS_EMPLOYED"] > 0, "DAYS_EMPLOYED"] = 0

    # Standardizing "SELLERPLACE_AREA" columns
    df.loc[df["SELLERPLACE_AREA"] < 0, "SELLERPLACE_AREA"] = 0
    return df


def days_transformer(X: pd.Series) -> pd.Series:
    """ Custom transformer function to process the days columns as follows:
        Standardizing Values:
        1. DAYS_EMPLOYED column has 18% positive values but this variable measured in negative. Hence, these positive values are outliers. \
            Dropping these values is not a good choice. So, convert these positive values to '0' before standardizing.
        2. Convert DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH, DAYS_LAST_PHONE CHANGE from negative to positive as days cannot be negative.
        3. Convert DAYS_BIRTH from negative to positive values and calculate age column.
        4. Convert All days columns to Years.
        5. Convert object datatype columns into categorical columns.
        6. Convert DAYS_DECISION from negative to positive values and create categorical bins columns.
    """

    transformed_data = X.copy()
    transformed_data = transformed_data.abs() // 365   # Convert negative values to positive and divide values by 365
    return transformed_data


def preprocess_data(df: pd.DataFrame, unwanted_columns: list, days_columns: list, scale_columns: list) -> Tuple[any, list]:
    """ Function to pre process the data and return pipeline job for downstream processes and transformed dataframe.
        Below is the procedure to apply the transformation:
        1. Drop the column if column exist in unwanted columns list.
        2. For numerical category type columns:

    """
    copy_df = df.copy()
    
    preprocessing_steps = []
    
    # Appending the transformer techniques for unwanted columns
    preprocessing_steps.append((f'drop_unwanted_columns', 'drop', unwanted_columns))        

    # Dropping the unwanted columns from duplicate dataframe to keep the accurate list of columns
    copy_df.drop(labels=unwanted_columns, axis=1, inplace=True)

    # Calculating the null value percentage for each column of the duplicate dataframe
    null_percentages = copy_df.isnull().mean() * 100.00

    # Appending the custom transformer techniques for days columns
    preprocessing_steps.append((f'days_transformer_{column}', FunctionTransformer(days_transformer), days_columns))
    
    # Appending the preprocessing steps for numerical columns
    for column in copy_df(include=['int', 'float']).columns:
        if column not in ['AMT_ANNUITY_y', 'AMT_GOODS_PRICE_y', 'CNT_PAYMENT']:
            if null_percentages[column] <= 15:
                # Apply median imputing technique to the column if null percentage is less than or equal to 15%
                preprocessing_steps.append((f'impute_{column}', SimpleImputer(strategy='median'), [column]))
            else:
                # Apply mean imputing technique to the column if null percentage is more than 15%
                preprocessing_steps.append((f'impute_{column}', SimpleImputer(strategy='mean'), [column]))

    # Appending the preprocessing steps for excluded numerical columns
    preprocessing_steps.append((f'impute_AMT_ANNUITY_y', SimpleImputer(strategy='median'), ['AMT_ANNUITY_y']))
    preprocessing_steps.append((f'impute_AMT_GOODS_PRICE_y', SimpleImputer(strategy='most_frequent'), ['AMT_GOODS_PRICE_y']))
    preprocessing_steps.append((f'impute_CNT_PAYMENT', SimpleImputer(strategy='constant', fill_value=0), ['CNT_PAYMENT']))

    # Appending the processing steps for applying standard scalar to selected numerical features
    preprocessing_steps.append((f'encode_StandardScaler_to_num', StandardScaler(), scale_columns))

    # Appending the preprocessing steps for categorical columns
    for column in copy_df(include='object').columns:
        if null_percentages[column] <= 15:
            # Apply mode imputing technique to the column if null percentage is less than or equal to 15%
            preprocessing_steps.append((f'impute_{column}', SimpleImputer(strategy='most_frequent'), [column]))
        else:
            # Impute with 'unknown' category if null percentage is more than 15%
            preprocessing_steps.append((f'impute_{column}', SimpleImputer(strategy='constant', fill_value='Unknown'), [column]))
        # Appending the preprocessing steps to apply the Label Encoding to the Categorical features
        preprocessing_steps.append((f'labelEncode_{column}', CategoricalEncoder(), [column]))

    # Create a pipline for the ColumnTransfer
    preprocessing_pipeline = ColumnTransformer(preprocessing_steps,n_jobs=-1, verbose=True)
  
    return preprocessing_pipeline, copy_df.columns