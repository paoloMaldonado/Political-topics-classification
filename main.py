from SVM import SVM_Model
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from vectorizer import SentenceEmbeddingVectorizer

# laod data
data = pd.read_pickle('../recollected data analysis/training data analysis/clean_full_dataset')

# label preprocessing
le = preprocessing.LabelEncoder()
le.fit(data.G3.to_list())

# create train/test split 
y = le.transform(data.G3.to_list())
X = data.tweet_clean.to_list()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

# # instantiate the SVM model
# model = SVM_Model(text_vectorization='tfidf')
# model.fit(X_train, y_train, fine_tune=True)
# y_pred = model.predict(X_test)

# print("test accuracy using best parameter values:", accuracy_score(y_test, y_pred))
# print("f1 macro using best parameter values:", f1_score(y_test, y_pred, average='macro'))
# print("f1 micro using best parameter values:", f1_score(y_test, y_pred, average='micro'))

# instantiate the SVM model with embeddings
model = SVM_Model(text_vectorization='word2vec', embedding_path="../embeddings")
model.fit(X_train, y_train, fine_tune=True)
y_pred = model.predict(X_test)

print("test data results")
print("-"*10)
print("test accuracy using best parameter values: {:0.4f}".format(accuracy_score(y_test, y_pred)))
print("f1 macro using best parameter values: {:0.4f}".format(f1_score(y_test, y_pred, average='macro')))
print("f1 micro using best parameter values: {:0.4f}".format(f1_score(y_test, y_pred, average='micro')))



