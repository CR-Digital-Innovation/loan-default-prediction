from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        #self.label_encoder = LabelEncoder()
        self.label_encoder.fit(X)
        return self

    def transform(self, X, y=None):
        X_encoded = self.label_encoder.transform(X)
        return X_encoded.reshape(-1, 1)
