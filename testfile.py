#from src import utils.S3Utils
from src.utils.s3_Functions import S3Utils
from src.utils.load_EnvVars import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_S3_BUCKET,
    AWS_S3_DATA_DIRECTORY,
    AWS_S3_DATA_DIRECTORY_MODELS,
    MODEL_NAME,
    MODEL_VERSION,
)


import sys
sys.path.append('src')

print(AWS_ACCESS_KEY_ID, AWS_S3_BUCKET)

s3_utils = S3Utils(
            AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET, AWS_S3_DATA_DIRECTORY
        )

print(s3_utils.check_file_exist(AWS_S3_DATA_DIRECTORY_MODELS, f"{MODEL_NAME}_{MODEL_VERSION}.pkl"))

model = s3_utils.load_pickle(AWS_S3_DATA_DIRECTORY_MODELS, f"{MODEL_NAME}_{MODEL_VERSION}.pkl", compressed=False)