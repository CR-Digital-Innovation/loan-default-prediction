
import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root directory to the PYTHONPATH
sys.path.insert(0, project_root)

# Importing required modules from utils
from utils.s3_Functions import S3Utils
from utils.load_EnvVars import (
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

        self.model = s3_utils.load_pickle(dirPath, fileName, compressed=True)

    def predict(self, data):
        return self.model.predict(data)
    
    def predict_proba(self, data):
        return self.model.predict_proba(data)
    
# Define a class for predicting through dataframes
class DataFramePayload(BaseModel):
    dataframe: List[Dict]
    id_process: int = Body(...)
    
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)
    
# Load the model
model_wrapper = ModelWrapper(AWS_S3_DATA_DIRECTORY_MODELS, f"{MODEL_NAME}_{MODEL_VERSION}.pkl")

@app.post("/csv_predict")
async def csv_predict(csv_data: list):
    try:
        dataframe = pd.DataFrame(csv_data)

        # Check if CSV file has at least one column
        if dataframe.columns.empty:
            return {"error": "The uploaded CSV file does not contain any columns."}
        else:
            if "TARGET" in dataframe.columns:
                dataframe.drop("TARGET", axis=1, inplace=True)
            
            dataframe = dataframe.replace(to_replace={None: np.nan})
            
            # Make predictions and capture prediction probabilities
            predictions = model_wrapper.predict(dataframe)
            probabilities = model_wrapper.predict_proba(dataframe)

            results = pd.DataFrame({
                "SK_ID_CURR": dataframe.iloc[:, 0],
                "Prediction Class": predictions,
                "Probability 0": probabilities[:, 0],
                "Probability 1": probabilities[:, 1]
            })

            return results.to_dict(orient='records')
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/predict")
async def predict(payload: DataFramePayload):
    try:
       
        dataframe = pd.DataFrame(payload.dataframe)

        # Check if CSV file has at least one column
        if dataframe.columns.empty:
            return {"error": "The uploaded CSV file does not contain any columns."}
        else:

            if "TARGET" in dataframe.columns:
                dataframe.drop("TARGET", axis=1, inplace=True)

            dataframe = dataframe.replace(to_replace={None: np.nan})

            predictions = model_wrapper.predict(dataframe)
            probabilities = model_wrapper.predict_proba(dataframe)

            results = pd.DataFrame({
                "SK_ID_CURR": dataframe.iloc[:, 0],
                "Prediction Class": predictions,
                "Probability 0": probabilities[:, 0],
                "Probability 1": probabilities[:, 1]
            })

            return results.to_dict(orient='records')
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
