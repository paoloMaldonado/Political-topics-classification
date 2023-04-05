from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, additional_features):
        self.additional_features = additional_features
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None, additional_features=None):
        if(additional_features == None):
            additional_features = self.additional_features 
        print("Adding additional features to the sentence matrix")
        try:
            sentence_matrix = np.concatenate((X, additional_features), axis=1)
        except ValueError:
            print("the sentence matrix and the additional feature matrix do not share the same number of rows")
        return sentence_matrix
