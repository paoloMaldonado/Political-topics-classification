from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.utils_embedding import checkOOVwords, createRandomOOV, wcbow, avgcbow
from utils.utils_tfidf import identity_tokenizer
from EmbeddingLoader import EmbeddingLoader


class SentenceEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, embedding, embedding_name, use_tfidf_weights=True, norm=False):
        self.use_tfidf_weights = use_tfidf_weights
        self.norm = norm
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
            oov_words_list = checkOOVwords(list(self.vocabulary), self.embedding)
            self.oov_to_vector = createRandomOOV(oov_words_list)
        return self
    
    def transform(self, X, y=None):
        # generate sentence embedding matrix
        if self.use_tfidf_weights:
            bow_array = self.bow_vectorizer.transform(X)
            return wcbow(X, self.vocabulary, bow_array.toarray(), self.embedding, self.oov_to_vector, self.norm)
        else:
            return avgcbow(X, self.embedding, self.oov_to_vector, self.norm)





