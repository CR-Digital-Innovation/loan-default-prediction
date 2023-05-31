"""Python file to split the data into train, test and cross validation data sets"""
import argparse
from sklearn.model_selection import train_test_split
from globalVars import AWS_S3_CLEAN_DATA_DIRECTORY
from dataFunctions import load_dataframe
from dataPreprocess import loan_process_df
from sklearn.preprocessing import StandardScaler


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

# Check if the --load option is provided
if args.load:
    # Check if filename is provided, otherwise use default
    filename = args.filename if args.filename else "clean_data.csv"
    filepath = AWS_S3_CLEAN_DATA_DIRECTORY + "/" + filename

    # Read CSV file into DataFrame
    clean_df = load_dataframe(filepath)
    print(f"DataFrame loaded from {filepath}:\n", clean_df.head())
else:
    clean_df = loan_process_df
    print("No load option provided. Loaded dataframe from dataprocessing job.")

# Split the data into features (X) and Target (y)
# X, y = loan_process_df.drop(columns="TARGET").values, loan_process_df["TARGET"].values
features, target = clean_df.drop(columns="TARGET"), loan_process_df["TARGET"]

# Scaling the training set before splitting the data
scaler = StandardScaler()
features[features.columns] = scaler.fit_transform(features[features.columns])

# Load X and y with values
X, y = features.values, target.values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=420
)

print("Shape of X_train and y_train:", X_train.shape, y_train.shape)
print("Shape of X_test and y_test:", X_test.shape, y_test.shape)

"""
# Split test data further into test and cross validaiation sets
X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=0.5, random_state=420
)
"""
