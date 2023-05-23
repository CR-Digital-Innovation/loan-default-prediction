import os
from dotenv import load_dotenv


# Load the variables from .env file
load_dotenv()

# Define the global variables to load AWS credentials and dataset file names
AWS_S3_BUCKET = os.environ.get('AWS_S3_BUCKET')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_S3_DATA_DIRECTORY = os.environ.get('AWS_S3_DATA_DIRECTORY')
DATASET_1 = os.environ.get('DATASET_1')
DATASET_2 = os.environ.get('DATASET_2')
