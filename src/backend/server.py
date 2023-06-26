"""FASTAPI Backend to make predictions"""
import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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
model_wrapper = ModelWrapper(
    AWS_S3_DATA_DIRECTORY_MODELS, f"{MODEL_NAME}_{MODEL_VERSION}.pkl"
)


def get_prediction(row):
    # Function to deterimine prediction category and confidence score
    class_0_prob = row["Probability 0"]
    class_1_prob = row["Probability 1"]

    if class_1_prob <= 0.2:
        prediction = "Confirmed Repayer"
        confidence_score = round(class_0_prob * 100, 2)
    elif class_1_prob < 0.7:
        prediction = "Probable Defaulter"
        confidence_score = round(max(class_0_prob, class_1_prob) * 100, 2)
    else:
        prediction = "Confirmed Defaulter"
        confidence_score = round(class_1_prob * 100, 2)

    return prediction, confidence_score


@app.post("/csv_predict")
async def csv_predict(csv_data: list):
    try:
        dataframe = pd.DataFrame(csv_data)

        # Check if CSV file has at least one column
        if dataframe.columns.empty:
            return {"error": "The uploaded CSV file does not contain any columns."}
        else:
            if "TARGET" in dataframe.columns:
                y_true = dataframe["TARGET"]
                dataframe.drop("TARGET", axis=1, inplace=True)

            dataframe = dataframe.replace(to_replace={None: np.nan})

            # Make predictions and capture prediction probabilities
            predictions = model_wrapper.predict(dataframe)
            probabilities = model_wrapper.predict_proba(dataframe)

            results = pd.DataFrame(
                {
                    "SK_ID_CURR": dataframe.iloc[:, 0],
                    "Prediction Class": predictions,
                    "Probability 0": probabilities[:, 0],
                    "Probability 1": probabilities[:, 1],
                }
            )

            # Apply transformation to create the new DataFrame
            results["Prediction"], results["Confidence Score in %"] = zip(
                *results.apply(get_prediction, axis=1)
            )

            # Select and reorder the desired columns
            predict_desc_df = results[["SK_ID_CURR", "Prediction", "Confidence Score in %"]]

            # Caluclate metrics
            cm = confusion_matrix(y_true, predictions)
            accuracy = accuracy_score(y_true, predictions)
            f1 = f1_score(y_true, predictions)

            return {
                "predictions": predict_desc_df.to_json(orient="records"),
                "cm": cm.tolist(),
                "accuracy": accuracy,
                "f1_score": f1,
            }
    except Exception as error:
        return {"error": str(error)}


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

            results = pd.DataFrame(
                {
                    "SK_ID_CURR": dataframe.iloc[:, 0],
                    "Prediction Class": predictions,
                    "Probability 0": probabilities[:, 0],
                    "Probability 1": probabilities[:, 1],
                }
            )

            return results.to_dict(orient="records")

    except Exception as error:
        return {"error": str(error)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
