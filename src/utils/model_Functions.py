""" Few useful functions for model training and evaluation"""

import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


def split_data(
    df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Function to split the data into train and test samples"""

    # Split the features and target variables
    X, y = df.drop(target_column, axis=1), df[target_column]

    # Split the data into training and test samples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X, y, X_train, X_test, y_train, y_test


def train_model(data_pipeline: any, X: pd.DataFrame, y: pd.DataFrame) -> Pipeline:
    """Function to preprocess the data followed by RandomForest Model training and return Pipeline."""

    # Create model Pipeline
    model_pipeline = Pipeline(
        [
            ("preprocessing", data_pipeline),  # Preprocessing pipeline
            ("model", RandomForestClassifier(n_jobs=-1)),  # Random Forest Classifier
        ]
    )

    # Fitting training data to the model pipeline
    model_pipeline.fit(X, y)

    return model_pipeline


def evaluate_model(model, X_test, y_test) -> None:
    """Function to evaluate the model on test data and print model metrics."""

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Print Classification report
    print("Classification report:\n", classification_report(y_test, y_pred))

    # Print Confusion Matrix
    print("Confusion Matrix", confusion_matrix(y_test, y_pred))
