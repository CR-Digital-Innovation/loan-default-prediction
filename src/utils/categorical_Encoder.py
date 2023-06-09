from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
        
    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders[col] = le
        return self
    
    def transform(self, X, y=None):
        X_encoded = X.copy()
        for col, le in self.label_encoders.items():
            X_encoded[col] = le.transform(X[col])
        return X_encoded