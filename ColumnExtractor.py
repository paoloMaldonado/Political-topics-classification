from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_extract='phrase'):
        self.column_to_extract = column_to_extract
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.column_to_extract == 'phrase':
            return X['phrase']
        elif self.column_to_extract == 'party':
            return X['party'].values.reshape(-1, 1)
        else:
            raise Exception("{} not defined".format(self.column_to_extract))
        # if(additional_features == None):
        #     additional_features = self.additional_features 
        # print("Adding additional features to the sentence matrix")
        # try:
        #     sentence_matrix = np.concatenate((X, additional_features), axis=1)
        # except ValueError:
        #     print("the sentence matrix and the additional feature matrix do not share the same number of rows")
        # return sentence_matrix
