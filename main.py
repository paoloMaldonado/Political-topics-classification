from SVM import SVM_Model
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from vectorizer import SentenceEmbeddingVectorizer

from utils.misc import removePunctuation, replaceNumbers, joinPhrases

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# load data
data = pd.read_pickle('dataset/manifesto_hispano_corpus.df')
data = data.head(10)

# label preprocessing
le = preprocessing.LabelEncoder()
le.fit(data.domain_name.values)

# create train/test split 
y = le.transform(data.domain_name.values).tolist()
X = {}
X['phrase'] = joinPhrases(data.prev_text.values, data.text.values)
X['party'] = data.partyname.values
# tokenize the sentences first
X['phrase'] = [replaceNumbers(removePunctuation(word_tokenize(sentence.lower(), language='spanish'))) for sentence in X['phrase']]
X = pd.DataFrame.from_dict(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

# instantiate the SVM model with embeddings
model = SVM_Model(text_vectorization='tfidf', embedding_path="../embeddings", additional_features=True)
model.fit(X_train, y_train, fine_tune=False, min_df=1, max_df=0.5, use_idf=True, C=1.0, gamma='scale', kernel='rbf')
y_pred = model.predict(X_test)



