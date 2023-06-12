from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
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

# Define the input data model
class InputData(BaseModel):
    data: pd.DataFrame = Field(..., description="Input data as a Pandas DataFrame")

    class Config:
        arbitrary_types_allowed = True

# Define a singleton model wrapper
class ModelWrapper:
    def __init__(self):
        self.model = None

    def load_model(self):
        """Load model from S3"""

        # Create an instance of S3Utils class to access various methods
        s3_utils = S3Utils(
            AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET, AWS_S3_DATA_DIRECTORY
        )

        # Defining the file name 
        filename = f"{MODEL_NAME}_{MODEL_VERSION}.pkl"

        # Loading the model from S3
        self.model = s3_utils.load_pickle(AWS_S3_DATA_DIRECTORY_MODELS, filename)

    def predict(self, data):
        return self.model.predict(data)

# Create the FastAPI app
app = FastAPI()

# Load the model
model_wrapper = ModelWrapper()
model_wrapper.load_model

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: InputData):
    # Extract the dataframe from the input data
    data = input_data.data

    # Make predictions using the loaded model
    predictions = model_wrapper.predict(data)

    # Return the predictions as a list
    return {"predictions": predictions.tolist()}
