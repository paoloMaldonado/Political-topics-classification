from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.utils_embedding import checkOOVwords, createRandomOOV, wcbow, avgcbow
from utils.utils_tfidf import identity_tokenizer
from EmbeddingLoader import EmbeddingLoader

class SentenceEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, embedding, embedding_name, use_tfidf_weights=True, norm=False, additional_features=None):
        self.use_tfidf_weights = use_tfidf_weights
        self.norm = norm
        self.additional_features = additional_features
        self.bow_vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False, use_idf=True, token_pattern=None)
        self.vocabulary = None
        self.embedding = embedding
        self.embedding_name = embedding_name

        self.oov_to_vector = None
        
    def fit(self, X, y=None):
        # generate bow array and vocabulary from tfidf vectorizer
        self.bow_vectorizer = self.bow_vectorizer.fit(X)
        self.vocabulary = self.bow_vectorizer.vocabulary_

        if self.embedding_name == 'word2vec':
            # generate oov vectors
            print("Checking for OOV words")
            oov_words_list = checkOOVwords(list(self.vocabulary), self.embedding)
            self.oov_to_vector = createRandomOOV(oov_words_list)
        return self
    
    def transform(self, X, y=None):
        # generate sentence embedding matrix
        print("Generating embedding vectors")
        if self.use_tfidf_weights:
            bow_array = self.bow_vectorizer.transform(X)
            sentence_matrix = wcbow(X, self.vocabulary, bow_array.toarray(), self.embedding, self.oov_to_vector, self.norm)
        else:
            sentence_matrix = avgcbow(X, self.embedding, self.oov_to_vector, self.norm)
        # if you want to add new features as a encoded vector (e.g categories, integers)
        # along the embedding matrix
        if self.additional_features != None:
            print("Adding additional features to the embedding matrix")
            try:
                sentence_matrix = np.concatenate((sentence_matrix, self.additional_features), axis=1)
            except ValueError:
                print("the embedding matrix and the additional feature matrix do not share the same number of rows")
        return sentence_matrix





