"""Model Training and evaluation Job"""

from utils.load_EnvVars import AWS_S3_DATA_DIRECTORY_MODELS
from dataPreprocess import applicationDf, unwanted_columns, scale_columns, s3_utils
from utils.model_Functions import split_data, train_model, evaluate_model
from utils.data_Functions import preprocess_data


# Split the data into train and test samples
X_train, X_test, y_train, y_test = split_data(applicationDf, "TARGET")

# Capture the data preprocessing parameters
preprocessing_pipeline, columns = preprocess_data(applicationDf, unwanted_columns, scale_columns)

# Train the model
model, accuracy = train_model(preprocessing_pipeline, X_train, y_train)
print(f"Best Model: {model},\nBest Score: {accuracy}")

# Evaluate the model
evaluate_model(model, X_test, y_test)

# Save the model
# Create an instance of S3Utils class to access various methods
s3_utils.save_pickle(
            AWS_S3_DATA_DIRECTORY_MODELS, "best_rf_model.pkl", model
        )