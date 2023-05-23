import pandas as pd
from dataIngestion import load_dataframe
from globalVars import DATASET_1


def null_value_df(df: pd.DataFrame) -> pd.DataFrame:
    """This function takes the dataframe as input and provides another dataframe with column names and percentage of null values in it"""

    null_values = round(df.isnull().sum() / df.shape[0] * 100.00, 2)
    null_df = pd.DataFrame(null_values.reset_index())
    null_df.columns = ['Column Name', 'Null values percentage']
    return null_df

def null_value_column_list(df: pd.DataFrame, percentage: float) -> list:
    """This function takes the dataframe, and float percentage value and provides the list of column names with greater than or equal to the defined percentage"""

    null_df = null_value_df(df)
    column_list = null_df[null_df['Null values percentage']>=percentage]['Column Name'].tolist()
    return column_list

applicationDf = load_dataframe(DATASET_1)
#previousDf = load_dataframe(DATASET_2)

#unwanted_columns_previousDf = null_value_column_list(previousDf, 40)

""" From EDA following decisions were made on application data.
    1. Columns with Null values greater than or equal to 40 percentage can be dropped.
    2. Regardless of amount of null values, EXT_SOURCE_X can be removed as they don't correlate with Target values.
    3. All FLAG_DOCUMENT_X columns except FLAG_DOCUMENT_3 can be deleted as submitting various documents does not influence the loan default rate.
    4. All columns related to contact parameters shall be dropped.
    """
unwanted_columns_applicationDf = null_value_column_list(applicationDf, 40)+['EXT_SOURCE_2', 'EXT_SOURCE_3', 'FLAG_DOCUMENT_2', 
        'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 
        'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 
        'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 
        'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']

applicationDf.drop(labels=unwanted_columns_applicationDf, axis=1, inplace=True)
print(applicationDf.shape[0], applicationDf.shape[1], len(unwanted_columns_applicationDf))