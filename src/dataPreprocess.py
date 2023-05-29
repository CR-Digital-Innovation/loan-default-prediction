"""Data Preparation functions to transform the data into required format for the Model training"""
import pandas as pd
from dataFunctions import (
    load_dataframe,
    null_value_column_list,
    convert_obj_to_cat,
    handle_null_values,
    label_encoding,
)
from sklearn.preprocessing import StandardScaler
from globalVars import DATASET_1, DATASET_2

# Load data into the dataframes
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


""" Standardizing Values:
    1. DAYS_EMPLOYED column has 18% positive values but this variable measured in negative. Hence, these positive values are outliers. \
        Dropping these values is not a good choice. So, convert these positive values to '0' before standardizing.
    2. Convert DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH from negative to positive as days cannot be negative.
    3. Convert DAYS_BIRTH from negative to positive values and calculate age and create categorical bins columns.
    4. Categorize the amount variables into bins.
    5. Categorize the Days of birth and employement into bins.
    6. Convert object datatype columns into categorical columns.
"""

# Standardizing DAYS_EMPLOYED values
applicationDf.loc[applicationDf["DAYS_EMPLOYED"] > 0, "DAYS_EMPLOYED"] = 0

# Converting Negative days to positive days
date_col = [
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
]

for col in date_col:
    applicationDf[col] = abs(applicationDf[col])
    # applicationDf[col] = applicationDf[col] //365

# Creating new column for Age
applicationDf["AGE"] = applicationDf["DAYS_BIRTH"] // 365

# Creating days columns to years
applicationDf["YEARS_EMPLOYED"] = applicationDf["DAYS_EMPLOYED"] // 365
applicationDf["YEARS_REGISTRATION"] = applicationDf["DAYS_REGISTRATION"] // 365
applicationDf["YEARS_ID_PUBLISH"] = applicationDf["DAYS_ID_PUBLISH"] // 365
applicationDf["YEARS_LAST_PHONE_CHANGE"] = (
    applicationDf["DAYS_LAST_PHONE_CHANGE"] // 365
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

unwanted_columns_applicationDf = (
    unwanted_columns_applicationDf
    + date_col
    + [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
    ]
)

# Dropping unwanted columns from application dataset.
applicationDf.drop(labels=unwanted_columns_applicationDf, axis=1, inplace=True)

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

# Standardizing "SELLERPLACE_AREA" columns
previousDf.loc[previousDf["SELLERPLACE_AREA"] < 0, "SELLERPLACE_AREA"] = 0

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

# Adding DAYS_DECISION to unwanted columns list
unwanted_columns_previousDf = unwanted_columns_previousDf + ["DAYS_DECISION"]

# Dropping unwanted columns from previous application dataset.
previousDf.drop(labels=unwanted_columns_previousDf, axis=1, inplace=True)

# Converting object datatype columns into categorical columns
previousDf = convert_obj_to_cat(previousDf)

# Imputing Null Values in applicationDf
applicationDf = handle_null_values(applicationDf)

# Imputing Null Values in previousDf
""" From EDA, we have learned that:
    1. AMT_ANNUITY column has many outliers. Hence, imputing with mean() is right approach.
    2. Upon analysing, median(), mode(), and mean() plots of AMT_GOODS_PRICE column w.r.t. to original data, imputing mode() resembles the plot of original data. Hence, mode() is applied.
    3. Impute CNT_PAYMENT with 0 as the NAME_CONTRACT_STATUS for these indicate that most of these loans were not started.
"""

previousDf["AMT_ANNUITY"].fillna(previousDf["AMT_ANNUITY"].median(), inplace=True)
previousDf["AMT_GOODS_PRICE"].fillna(
    previousDf["AMT_GOODS_PRICE"].mode()[0], inplace=True
)
previousDf["CNT_PAYMENT"].fillna(0, inplace=True)

# Merge DataSet
loan_process_df = pd.merge(applicationDf, previousDf, how="inner", on="SK_ID_CURR")
# print("Shape of merged dataset:", loan_process_df.shape)

# Convert categorical columns into numerical columns
loan_process_df = label_encoding(loan_process_df)

# Dropping SK_ID_CURR and SK_ID_PREV as they do not influence the Target variable
loan_process_df.drop(labels=["SK_ID_CURR", "SK_ID_PREV"], axis=1, inplace=True)

# Listing columns with unique values more than 500
unique = loan_process_df.nunique()

uniqueDf = pd.DataFrame(unique.reset_index())
uniqueDf.columns = ["Column Name", "Unique Values"]
unique_columns = uniqueDf[uniqueDf["Unique Values"] >= 500]["Column Name"].tolist()

# Applying Standard Scalar to the unique columns
scaler = StandardScaler()
loan_process_df[unique_columns] = scaler.fit_transform(loan_process_df[unique_columns])
