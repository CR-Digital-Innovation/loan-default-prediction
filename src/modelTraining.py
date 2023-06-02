"""Python file to load and split the data into train, test sets and also to try train, evaluate and save the trained model"""
import argparse
from dataFunctions import load_clean_data_s3
from modelFunctions import data_split, train_model, evaluate_model, save_model

""" Provide --load flag to load the cleaned data csv as dataframe from s3 file store else it loads dataframe by processing the dataPreprocess job.
    Optionally, you can provide a file name with --filename argument.
    Ex:
        python dataPreparation.py --load --filename custom_data.csv
"""
parser = argparse.ArgumentParser(
    description="Read Clean data CSV file from s3 file storage"
)
parser.add_argument("--load", action="store_true", help="Load CSV file into DataFrame")
parser.add_argument("--filename", type=str, help="Specify the filename (optional)")

# Parse the command-line arguments
args = parser.parse_args()

# Load the clean data from the s3 file storage
clean_df = load_clean_data_s3(load=args.load, filename=args.filename)

# Split the data into train and test samples
X_train, X_test, y_train, y_test = data_split(clean_df, "TARGET")

# Train the model
best_model, best_score = train_model(X_train, y_train)
print(f"Best Model: {best_model},\nBest Score: {best_score}")

# Evaluate the model
evaluate_model(best_model, X_test, y_test)

# Save the model
save_model(best_model, "best_rf_model.pkl")
