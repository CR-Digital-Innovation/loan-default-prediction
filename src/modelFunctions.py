import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def data_split(
    df: pd.Dataframe, target_column: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Function to split the data into train and test samples"""

    # Split the features and target variables
    X, y = df.drop(target_column, axis=1), df[target_column]

    # Split the data into training and test samples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def train_model(X, y):
    """Function to train the Random Forest Classifier model with cleaned data using Grid Search Cross Validation to find the best model and accuracy."""

    # Create a Pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),  # Apply StandardScaler
            ("classifier", RandomForestClassifier()),  # Random Forest Classifier
        ]
    )

    # Define hyperparametes for tuning the model
    param_grid = {
        "max_depth": [None, 10, 20, 50, 100],  # Maximum number of levels in tree
        "min_samples_split": [
            2,
            5,
            10,
        ],  # Minimum number of samples required to split an internal node
        "min_samples_leaf": [
            1,
            2,
            4,
        ],  # Minimum number of samples required at each leaf node
        "n_estimators": [50, 100, 200, 500],  # Number of trees in random forest
        "n_jobs": -1,
    }

    # Performing grid search cross-validation
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
    grid_search.fit(X, y)

    # Get the best model and it's score
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_

    # Print best score and best model
    # print(f"Best Model: {best_model}, Best Score: {best_score}")

    return best_model, best_score


def evaluate_model(model, X_test, y_test):
    """Function to evaluate the model on test data and print model metrics."""

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Print Classification report
    print("Classification report:\n", classification_report(y_test, y_pred))

    # Display COnfusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    disp.plot()
    plt.show()


def save_model(model, filename: str):
    """Function to save the model to pickle file."""

    # Check if models directory exists, create if not
    model_dir = os.path.join("models")
    os.makedirs(model_dir, exist_ok=True)
    model_file_path = os.path.join(model_dir, filename)

    with open(model_file_path, "wb") as file:
        pickle.dump(model, file)


def load_model(filename: str):
    """Function to load the model from pickle file."""

    with open(filename, "rb") as file:
        loaded_model = pickle.load(file)

    return loaded_model
