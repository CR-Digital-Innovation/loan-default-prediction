import pandas as pd
from globalVars import AWS_S3_BUCKET, AWS_S3_DATA_DIRECTORY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY


def load_dataframe(fileName: str) -> pd.DataFrame:   
    """Loading csv file that is present in AWS S3 Bucket as dataframe using Pandas """

    dataframe = pd.read_csv(
        f"s3://{AWS_S3_BUCKET}/{AWS_S3_DATA_DIRECTORY}/{fileName}",
        storage_options={
            "key": AWS_ACCESS_KEY_ID,
            "secret": AWS_SECRET_ACCESS_KEY,
        },
    )
    return dataframe
