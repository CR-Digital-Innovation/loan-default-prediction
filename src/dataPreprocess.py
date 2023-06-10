""" Data Preprocess job"""

import pandas as pd
from utils.load_EnvVars import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_S3_BUCKET,
    AWS_S3_DATA_DIRECTORY,
    AWS_S3_DATA_DIRECTORY_RAW,
    AWS_S3_DATA_DIRECTORY_PROCESSED,
    APPLICATION_DATASET,
)
from utils.s3_Functions import S3Utils
from utils.data_Functions import (
    null_value_column_list,
    custom_standardization,
    preprocess_data,
)


# Create an instance of S3Utils class to access various methods
s3_utils = S3Utils(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET, AWS_S3_DATA_DIRECTORY)

# Load data into a dataframe
applicationDf = s3_utils.load_dataframe(AWS_S3_DATA_DIRECTORY_RAW, APPLICATION_DATASET)

""" From EDA following decisions were made on application data. \
    1. Columns with Null values greater than or equal to 40 percentage can be dropped.
    2. Regardless of amount of null values, EXT_SOURCE_X can be removed as they don't correlate \
        with Target values.
    3. All FLAG_DOCUMENT_X columns except FLAG_DOCUMENT_3 can be deleted as submitting various \
        documents does not influence the loan default rate.
    4. All columns related to contact parameters shall be dropped.
    5. Drop unnecessary columns such as 'WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START', \
        'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY'.
"""
unwanted_columns = null_value_column_list(applicationDf, 35) + [
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
    "WEEKDAY_APPR_PROCESS_START_x",
    "HOUR_APPR_PROCESS_START_x",
    "WEEKDAY_APPR_PROCESS_START_y",
    "HOUR_APPR_PROCESS_START_y",
    "FLAG_LAST_APPL_PER_CONTRACT",
    "NFLAG_LAST_APPL_IN_DAY",
    "SK_ID_CURR",
    "SK_ID_PREV",
]

scale_columns = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT_x",
    "AMT_ANNUITY_x",
    "AMT_GOODS_PRICE_x",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "CNT_FAM_MEMBERS",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "AMT_ANNUITY_y",
    "AMT_APPLICATION",
    "AMT_CREDIT_y",
    "AMT_GOODS_PRICE_y",
    "SELLERPLACE_AREA",
    "CNT_PAYMENT",
]

if __name__ == '__main__':
    """ Save processed dataframe to either csv or as a pickle file when needed

        To save the processed dataframe as csv file:
            1. Provide --save-to-csv flag.
            2. Optionally, you can parse a filename with --filename argument.
            Ex:
                python dataPreprrocess.py --save-to-csv --filename 'processed_data'

        To save the processed dataframe as a pickle file:
            1. Provide --save-to-pkl flag.
            2. Optionally, you can parse a filename with --filename argument.
            Ex:
                python dataPreprocess.py --save-to-pkl --filename 'processed_data'
    """

    import argparse

    parser = argparse.ArgumentParser(description="Save the processed dataframe to S3 storage")
    parser.add_argument("--save-to-csv", action="store_true", help="Save DataFrame as CSV")
    parser.add_argument("--save-to-pkl", action="store_true", help="Save DataFrame as Pickle file")
    parser.add_argument("--filename", type=str, help="Specify the filename (optional)")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Apply custom standardizations to the desired columns
    applicationDf = custom_standardization(applicationDf)

    # Execute the data preprocess pipeline to return the data preprocess pipeline binary and list of remaining columns
    preprocessing_pipeline, columns = preprocess_data(applicationDf, unwanted_columns, date_columns, scale_columns)

    # Apply the preprocessing pipeline to the dataframe
    data_transform_pipeline = preprocessing_pipeline.fit_transform(applicationDf)

    # Create a DataFrame from the transformed data
    transformed_df = pd.DataFrame(
        data_transform_pipeline,
        columns=columns
    )

    if args.save_to_csv:
        # Check if filename is provided, otherwise use default
        filename = f"{args.filename}.csv" if args.filename else "processed_data.csv"

        # Save dataframe as CSV file
        s3_utils.save_dataframe(AWS_S3_DATA_DIRECTORY_PROCESSED, filename, transformed_df)
    
    if args.save_to_pkl:
        # Check if filename is provided, otherwise use default
        filename = f"{args.filename}.pkl" if args.filename else "processed_data.pkl"

        # Save dataframe as Pickle file
        s3_utils.save_pickle(AWS_S3_DATA_DIRECTORY_PROCESSED, filename, data_transform_pipeline)
