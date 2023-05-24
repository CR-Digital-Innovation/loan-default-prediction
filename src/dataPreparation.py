"""Data Preparation functions to transform the data into required format for the Model training"""
import pandas as pd
from dataIngestion import load_dataframe
from globalVars import DATASET_1, DATASET_2


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


applicationDf = load_dataframe(DATASET_1)
previousDf = load_dataframe(DATASET_2)

""" From EDA following decisions were made on application data. \
    1. Columns with Null values greater than or equal to 40 percentage can be dropped.
    2. Regardless of amount of null values, EXT_SOURCE_X can be removed as they don't correlate \
        with Target values.
    3. All FLAG_DOCUMENT_X columns except FLAG_DOCUMENT_3 can be deleted as submitting various \
        documents does not influence the loan default rate.
    4. All columns related to contact parameters shall be dropped.
"""
unwanted_columns_applicationDf = null_value_column_list(applicationDf, 40) + [
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "FLAG_DOCUMENT_2",
    "FLAG_DOCUMENT_4",
    "FLAG_DOCUMENT_5",
    "FLAG_DOCUMENT_6",
    "FLAG_DOCUMENT_7",
    "FLAG_DOCUMENT_8",
    "FLAG_DOCUMENT_9",
    "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_11",
    "FLAG_DOCUMENT_12",
    "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_14",
    "FLAG_DOCUMENT_15",
    "FLAG_DOCUMENT_16",
    "FLAG_DOCUMENT_17",
    "FLAG_DOCUMENT_18",
    "FLAG_DOCUMENT_19",
    "FLAG_DOCUMENT_20",
    "FLAG_DOCUMENT_21",
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
]

# Dropping unwanted columns from application dataset.
applicationDf.drop(labels=unwanted_columns_applicationDf, axis=1, inplace=True)

""" Standardizing Values:
    1. Convert DAYS_EMPLOYED, DAYS_REGISTRATION,DAYS_ID_PUBLISH from negative to positive as days cannot be negative.
    2. Convert DAYS_BIRTH from negative to positive values and calculate age and create categorical bins columns.
    3. Categorize the amount variables into bins.
    4. Categorize the Days of birth and employement into bins.
    5. Convert object datatype columns into categorical columns.
"""

# Converting Negative days to positive days
date_col = ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"]

for col in date_col:
    applicationDf[col] = abs(applicationDf[col])

# Creating bins for Age
applicationDf["AGE"] = applicationDf["DAYS_BIRTH"] // 365
bins = [0, 20, 30, 40, 50, 100]
slots = ["0-20", "20-30", "30-40", "40-50", "50 above"]

applicationDf["AGE_GROUP"] = pd.cut(applicationDf["AGE"], bins=bins, labels=slots)

# Creating bins for Employement Time
applicationDf["YEARS_EMPLOYED"] = applicationDf["DAYS_EMPLOYED"] // 365
bins = [0, 5, 10, 20, 30, 40, 50, 60, 150]
slots = ["0-5", "5-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60 above"]

applicationDf["EMPLOYMENT_YEAR"] = pd.cut(
    applicationDf["YEARS_EMPLOYED"], bins=bins, labels=slots
)

# Creating bins for income amount
applicationDf["AMT_INCOME_TOTAL"] = applicationDf["AMT_INCOME_TOTAL"] / 100000
bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
slots = [
    "0-100K",
    "100K-200K",
    "200k-300k",
    "300k-400k",
    "400k-500k",
    "500k-600k",
    "600k-700k",
    "700k-800k",
    "800k-900k",
    "900k-1M",
    "1M Above",
]

applicationDf["AMT_INCOME_RANGE"] = pd.cut(
    applicationDf["AMT_INCOME_TOTAL"], bins, labels=slots
)

# Creating bins for Credit amount
applicationDf["AMT_CREDIT"] = applicationDf["AMT_CREDIT"] / 100000

bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
slots = [
    "0-100K",
    "100K-200K",
    "200k-300k",
    "300k-400k",
    "400k-500k",
    "500k-600k",
    "600k-700k",
    "700k-800k",
    "800k-900k",
    "900k-1M",
    "1M Above",
]

applicationDf["AMT_CREDIT_RANGE"] = pd.cut(
    applicationDf["AMT_CREDIT"], bins=bins, labels=slots
)

# Converting object datatype columns into categorical columns
applicationDf = convert_obj_to_cat(applicationDf)

""" From EDA following decisions were made on previous application data.
    1. Columns with Null values greater than or equal to 40 percentage can be dropped.
    2. Drop unnecessary columns such as 'WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START', \
        'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY'.
    3. Convert DAYS_DECISION from negative to positive values and create categorical bins columns.
    4. Convert object datatype columns into categorical columns.
"""
unwanted_columns_previousDf = null_value_column_list(previousDf, 40) + [
    "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START",
    "FLAG_LAST_APPL_PER_CONTRACT",
    "NFLAG_LAST_APPL_IN_DAY",
]

# Dropping unwanted columns from previous application dataset.
previousDf.drop(labels=unwanted_columns_previousDf, axis=1, inplace=True)

# Converting negative days to positive days
previousDf["DAYS_DECISION"] = abs(previousDf["DAYS_DECISION"])

# age group calculation e.g. 388 will be grouped as 300-400
previousDf["DAYS_DECISION_GROUP"] = (
    (previousDf["DAYS_DECISION"] - (previousDf["DAYS_DECISION"] % 400)).astype(str)
    + "-"
    + (
        (previousDf["DAYS_DECISION"] - (previousDf["DAYS_DECISION"] % 400))
        + (previousDf["DAYS_DECISION"] % 400)
        + (400 - (previousDf["DAYS_DECISION"] % 400))
    ).astype(str)
)

# Converting object datatype columns into categorical columns
previousDf = convert_obj_to_cat(previousDf)
