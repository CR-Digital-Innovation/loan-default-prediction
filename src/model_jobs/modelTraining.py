# pylint: disable=W0105
"""Model Training and evaluation Job"""

""" import os
import sys

# Get the root directory of your project
root_directory = os.path.dirname(os.path.abspath(__file__))

# Add the root directory to the Python path
sys.path.append(root_directory)
print(root_directory)
 """

import os
import sys

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root directory to the PYTHONPATH
sys.path.insert(0, project_root)

from utils.load_EnvVars import AWS_S3_DATA_DIRECTORY_MODELS, MODEL_NAME, MODEL_VERSION
from dataPreprocess import applicationDf, unwanted_columns, scale_columns, s3_utils
from utils.model_Functions import split_data, train_model, evaluate_model
from utils.data_Functions import preprocess_data


# Split the data into train and test samples
print("---------------------- SPLITTING TRAIN AND TEST SAMPLES ----------------------")
X, y, X_train, X_test, y_train, y_test = split_data(applicationDf, "TARGET")

# Capture the data preprocessing parameters
print("---------------------- DATA PREPROCESSING ----------------------")
preprocessing_pipeline, columns = preprocess_data(X, unwanted_columns, scale_columns)

# Train the model
print("---------------------- MODEL TRAINING ----------------------")
model= train_model(preprocessing_pipeline, X_train, y_train)

# Evaluate the model
print("---------------------- MODEL EVALUATING ----------------------")
evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    """Save the model to S3 with your desired file name else will be saved with default name 

    To save the model as a pickle file with your choice of file name provide the --filename flag followed by your filename
        Ex:
            python modelTraining.py --filename 'processed_data'
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="Save the model to S3 storage with cutsom filename"
    )
    parser.add_argument("--filename", type=str, help="Specify the filename (optional)")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if filename is provided, otherwise use default
    filename = f"{args.filename}.pkl" if args.filename else f"{MODEL_NAME}_{MODEL_VERSION}.pkl"

    # Save the model
    print("---------------------- SAVING THE MODEL TO S3 ----------------------")
    s3_utils.save_pickle(
                AWS_S3_DATA_DIRECTORY_MODELS, filename, model,
            )
