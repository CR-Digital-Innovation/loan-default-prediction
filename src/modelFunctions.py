import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def model_train



# Load DataFrame
df = pd.read_csv('data.csv')  # Replace 'data.csv' with your file path

# Separate features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Apply StandardScaler
    ('classifier', RandomForestClassifier())  # Random Forest Classifier
])

# Define hyperparameters for tuning
param_grid = {
    'classifier__n_estimators': [100, 200, 300],  # Number of trees in the forest
    'classifier__max_depth': [None, 5, 10],  # Maximum depth of the tree
    'classifier__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
}

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform grid search cross-validation
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model and its score
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_
print(f'Best Model: {best_model}')
print(f'Best Score: {best_score}')

# Predict on the testing data using the best model
y_pred = best_model.predict(X_test)

# Evaluate the model (e.g., calculate accuracy, f1-score, etc.)
accuracy = best_model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

# Additional steps or analysis with the best model can be added here
