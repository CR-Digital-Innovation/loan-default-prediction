"""Few useful data preprocessing functions"""

import pandas as pd
from typing import Tuple
from categorical_Encoder import CategoricalEncoder # Custom LabelEncoder() technique through ColumnTransfer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline


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


def days_employed_standardization(X: pd.Series) -> pd.Series:
    """Function to transform the 'DAYS_EMPLOYED' column values"""

    transformed_data = X.copy()
    transformed_data[transformed_data > 0] = 0  # Standardizing DAYS_EMPLOYED values
    return transformed_data


def sellerplace_area_standardization(X: pd.Series) -> pd.Series:
    """Function to transform the 'SESELLERPLACE_AREA' column values"""

    transformed_data = X.copy()
    transformed_data[transformed_data < 0] = 0  # Standardizing "SELLERPLACE_AREA" columns
    return transformed_data


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


def preprocess_data(df: pd.DataFrame, unwanted_columns: list, scale_columns: list) -> Tuple[any, list]:
    """ Function to pre process the data and return pipeline job for downstream processes and transformed dataframe.
        Below is the procedure to apply the transformation:
        1. Drop the unwanted columns.
        2. Apply any custom transformations if exist.
        3. For numerical category type columns: 
            A. If null value percentage is less than or equal to 15% then median() is used.
            B. If null value is above 15% then mean() is used.
            C. Apply StandardScaler() to the selected columns.
        4. For Categorical type columns:
            A. If null value percentage is less than or equal to 15% then mode() is applied.
            B. If null value is above 15% then new "Unknown" category is created.
            C. Convert Categorical features into numerical features.
    """
    copy_df = df.copy()
    
    preprocessing_steps = []
    
    # Appending the transformer techniques for unwanted columns
    preprocessing_steps.append((f'drop_unwanted_columns', 'drop', unwanted_columns))        

    # Dropping the unwanted columns from duplicate dataframe to keep the accurate list of columns
    copy_df.drop(labels=unwanted_columns, axis=1, inplace=True)

    # Making list of columns with special transformations
    days_columns = [
        "DAYS_BIRTH",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE",
        "DAYS_DECISION",
    ]
    special_columns = ['DAYS_EMPLOYED','AMT_ANNUITY_y', 'AMT_GOODS_PRICE_y', 'CNT_PAYMENT', 'SELLERPLACE_AREA']

    # Calculating the null value percentage for each column of the duplicate dataframe
    null_percentages = copy_df.isnull().mean() * 100.00

    # Appending the preprocessing steps for numerical columns

    # Custom Pipelines for median and mean transformers 
    Impute_median_scale = Pipeline([
                    (f'impute_{column}', SimpleImputer(strategy='median')),
                    (f'scale_{column}', StandardScaler())
                ])
    Impute_mean_scale = Pipeline([
                    (f'impute_{column}', SimpleImputer(strategy='mean')),
                    (f'scale_{column}', StandardScaler())
                ])

    for column in copy_df.select_dtypes(include=['int', 'float']).columns:
        if column in days_columns:
            if column in scale_columns:
                # Apply custom function transformer and standard scaling
                # No need to Impute Days columns as they have no missing values
                preprocessing_steps.append(f'transform_{column}', Pipeline([
                        (f'days_transform_{column}', FunctionTransformer(days_transformer)),
                        (f'scale_{column}', StandardScaler())
                    ]), [column])
            else:
                # Apply custom function transformer only
                # No need to Impute Days columns as they have no missing values
                preprocessing_steps.append(f'days_transform_{column}', FunctionTransformer(days_transformer), [column])
        elif column in scale_columns and column not in special_columns:
            if null_percentages[column] <= 15:
                # Apply median imputing technique to the column if null percentage is less than or equal to 15%
                # Also applying standard scalar to selected numerical features
                preprocessing_steps.append(('numerical_tranformations', Impute_median_scale, [column]))
            else:
                # Apply mean imputing technique to the column if null percentage is more than 15%
                # Also applying standard scalar to selected numerical features
                preprocessing_steps.append(('numerical_tranformations', Impute_mean_scale, [column]))
        elif column not in special_columns:
            if null_percentages[column] <= 15:
                # Apply median imputing technique to the column if null percentage is less than or equal to 15%
                # Also applying standard scalar to selected numerical features
                preprocessing_steps.append(('numerical_tranformations', Impute_median_scale, [column]))
            else:
                # Apply mean imputing technique to the column if null percentage is more than 15%
                # Also applying standard scalar to selected numerical features
                preprocessing_steps.append(('numerical_tranformations', Impute_mean_scale, [column]))

    # Appending the preprocessing steps for special numerical columns
    preprocessing_steps.append((f'transform_DAYS_EMPLOYED', Pipeline([
            (f'standardize_DAYS_EMPLOYED', FunctionTransformer(days_employed_standardization)),
            (f'days_transform_DAYS_EMPLOYED', FunctionTransformer(days_transformer)),
            (f'scale_DAYS_EMPLOYED', StandardScaler())
        ])), ['DAYS_EMPLOYED'])

    preprocessing_steps.append((f'transform_AMT_ANNUITY_y', Pipeline([
            (f'impute_AMT_ANNUITY_y', SimpleImputer(strategy='median')),
            (f'scale_AMT_ANNUITY_y', StandardScaler())
        ]), ['AMT_ANNUITY_y']))

    preprocessing_steps.append((f'transform_AMT_GOODS_PRICE_y', Pipeline([
            (f'impute_AMT_GOODS_PRICE_y', SimpleImputer(strategy='most_frequent')),
            (f'scale_AMT_GOODS_PRICE_y', StandardScaler())
        ]), ['AMT_GOODS_PRICE_y']))

    preprocessing_steps.append((f'transform_CNT_PAYMENT', Pipeline([
            (f'impute_CNT_PAYMENT', SimpleImputer(strategy='constant', fill_value=0)),
            (f'scale_CNT_PAYMENT', StandardScaler())
        ]), ['CNT_PAYMENT']))

    preprocessing_steps.append((f'transform_SELLERPLACE_AREA', Pipeline([
            (f'standardize_SELLERPLACE_AREA', FunctionTransformer(sellerplace_area_standardization)), 
            (f'scale_SELLERPLACE_AREA', StandardScaler())
        ]), ['SELLERPLACE_AREA']))

    # Appending the preprocessing steps for categorical columns
    for column in copy_df.select_dtypes(include='object').columns:
        if null_percentages[column] <= 15:
            # Apply mode imputing technique to the column if null percentage is less than or equal to 15%
            # Also pply the Label Encoding to the Categorical features
            preprocessing_steps.append(('categorical_tranformations', Pipeline([
                    (f'impute_{column}', SimpleImputer(strategy='most_frequent')),
                    (f'encode_{column}', CategoricalEncoder())
            ]), [column]))
        else:
            # Impute with 'unknown' category if null percentage is more than 15%
            # Also pply the Label Encoding to the Categorical features
            preprocessing_steps.append(('categorical_tranformations', Pipeline([
                    (f'impute_{column}', SimpleImputer(strategy='constant', fill_value='Unknown')),
                    (f'encode_{column}', CategoricalEncoder())
            ]), [column]))

    # Create a pipline for the ColumnTransfer
    preprocessing_pipeline = ColumnTransformer(preprocessing_steps, n_jobs=-1, verbose=True)
  
    return preprocessing_pipeline, copy_df.columns