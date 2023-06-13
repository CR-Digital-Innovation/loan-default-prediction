# pylint: disable=W0105
"""FAST API modules to load the pickle file and make predictions"""

# Append the root path to the src folder, so that detect the utils modules
import sys
sys.path.append('src')

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

    def predict(self, data):
        return self.model.predict(data)
    
    def predict_proba(self, data):
        return self.model.predict_proba(data)

# Create the FastAPI app
app = FastAPI(swagger_ui_parameters={'syntaxHighlight': False})

# Load the model
model_wrapper = ModelWrapper(AWS_S3_DATA_DIRECTORY_MODELS, f"{MODEL_NAME}_{MODEL_VERSION}.pkl")

# Define the prediction endpoint
@app.post("/csv-predict")
async def create_upload_file(file: UploadFile = File(...)):
    # Handle the file only if it is a CSV
    if file.filename.endswith(".csv"):
        # Create a temporary file with the same name as the uploaded 
        # CSV file to load the data into a pandas Dataframe
        with open(file.filename, "wb")as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)
        if "TARGET" in data.columns:
            data.drop("TARGET", axis=1, inplace=True)

        # Make predictions and determine prediction probabilities
        predictions = model_wrapper.predict(data)
        prediction_probabilities = model_wrapper.predict_proba(data)

        # Return a JSON object containing the model predictions
        return {
            "Predictions": predictions.tolist(),
            "Prediction_Probabilities": prediction_probabilities.tolist()
        }    
    else:
        # Raise a HTTP 400 Exception, indicating Bad Request 
        # (you can learn more about HTTP response status codes here)
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")
