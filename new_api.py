from fastapi import FastAPI
from fastapi import Request
from pydantic import BaseModel


import sys
import os
import pandas as pd

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
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

# Append the root path to the src folder, so that detect the utils modules
sys.path.append('src')

# Create an instance of S3Utils class to access various methods
s3_utils = S3Utils(
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET, AWS_S3_DATA_DIRECTORY
)

# Define a singleton model wrapper
class ModelWrapper:
    def __init__(self, dirPath: str, fileName: str):
        """Load model from S3"""

        # Defining the file name
        #filename = f"{MODEL_NAME}_{MODEL_VERSION}.pkl"

        # Loading the model from S3
        self.model = s3_utils.load_pickle(dirPath, fileName, compressed=True)

app = FastAPI()

# Load the model
model_wrapper = ModelWrapper(AWS_S3_DATA_DIRECTORY_MODELS, f"{MODEL_NAME}_{MODEL_VERSION}.pkl")


class Data(BaseModel):
    dataframe: dict


@app.post("/predict")
async def predict(data: Data):
    try:
        dataframe = pd.DataFrame(data.dataframe)

        if "TARGET" in dataframe.columns:
            dataframe.drop("TARGET", axis=1, inplace=True)

        predictions = ModelWrapper.predict(dataframe)
        probabilities = ModelWrapper.predict_proba(dataframe)

        results = pd.DataFrame({
            "SK_ID_CURR": dataframe.iloc[:, 0],
            "Prediction Class": predictions,
            "Probability 0": probabilities[:, 0],
            "Probability 1": probabilities[:, 1]
        })

        return results.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}
