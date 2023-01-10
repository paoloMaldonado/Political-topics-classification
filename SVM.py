import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from utils.utils_tfidf import identity_tokenizer
from vectorizer import SentenceEmbeddingVectorizer
from gensim.models import KeyedVectors
from EmbeddingLoader import EmbeddingLoader

class SVM_Model:
    def __init__(self, text_vectorization, embedding_path):
        self.text_vectorization = text_vectorization
        self.model = None
        self.embedding_path = embedding_path
        if text_vectorization != "tfidf":
            embLoad = EmbeddingLoader(self.text_vectorization, self.embedding_path).load_embedding_model()
            self.embedding = embLoad.embedding_object
    
    def fit_and_optimize(self, X, y):
        if self.text_vectorization == 'tfidf':
            print("fine-tunning SVM +", self.text_vectorization)
            objs = [("tfidf", TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)),
                    ("svm", SVC(kernel="rbf"))]
            pipe = Pipeline(objs)

            # Specify parameters of the pipeline and their ranges for grid search
            params = {
                'tfidf__min_df': np.linspace(0.005, 0.1, 5),
                'tfidf__max_df': np.linspace(0.7, 1.0, 4),
                'tfidf__use_idf': (True, False),
                'svm__C': np.logspace(-1, 2, 10),
                'svm__gamma': np.logspace(-1, 1, 10),
                'svm__kernel': ('linear', 'rbf')
            }

            # Construct our grid search object
            search = GridSearchCV(pipe, param_grid=params, verbose=2, n_jobs=-1)
            return search.fit(X, y)

        else:
            print("fine-tunning SVM +", self.text_vectorization)
            objs = [("embedding", SentenceEmbeddingVectorizer(self.embedding, self.text_vectorization)),
                    ("svm", SVC(kernel="rbf"))]
            pipe = Pipeline(objs)

            params = [{"embedding__use_tfidf_weights": [True, False],
                       'svm__C': np.logspace(-1, 2, 10),
                       'svm__gamma': np.logspace(-1, 1, 10)
                       }]

            search = GridSearchCV(pipe, param_grid=params, verbose=2, n_jobs=2)
            return search.fit(X, y)


    def fit(self, X, y, fine_tune=True, **kwargs):
        if self.text_vectorization == 'tfidf' and fine_tune == False:
            if len(kwargs) == 0:
                print("Training SVM + {} model with default parameters".format(self.text_vectorization))
                objs = [("tfidf", TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)),
                        ("svm", SVC(kernel="rbf"))]
            else:
                print("Training SVM + {} model with custom parameters".format(self.text_vectorization))
                objs = [("tfidf", TfidfVectorizer(tokenizer=identity_tokenizer, 
                                                  lowercase=False, 
                                                  min_df=kwargs['min_df'], 
                                                  max_df=kwargs['max_df'],
                                                  use_idf=kwargs['use_idf'])),
                        ("svm", SVC(C=kwargs['C'],
                                    gamma=kwargs['gamma'],
                                    kernel=kwargs['kernel']))]
            pipe = Pipeline(objs)
            self.model = pipe.fit(X, y)
            print("Training complete")

        elif self.text_vectorization == 'tfidf' and fine_tune == True:
            print("Training and fine tunning the model")
            self.model = self.fit_and_optimize(X, y)

            print("Fine tunning metrics:")
            print("-"*10)
            print("CV Score using best parameter values:", self.model.best_score_)
            print("Best parameter values:")
            for param in self.model.best_params_.items():
                print(param)
            print("-"*10)

            print("Training and fine tunning complete")

        elif self.text_vectorization != 'tfidf' and fine_tune == False:
            if len(kwargs) == 0:
                print("Training SVM + {} model with default parameters".format(self.text_vectorization))
                objs = [("embedding", SentenceEmbeddingVectorizer(self.embedding, self.text_vectorization)),
                        ("svm", SVC(kernel="rbf"))]
            else:
                print("Training SVM + {} model with custom parameters".format(self.text_vectorization))
                objs = [("embedding", SentenceEmbeddingVectorizer(embedding=self.embedding,
                                                                  embedding_name=self.text_vectorization,
                                                                  use_tfidf_weights=kwargs['use_tfidf_weights'],
                                                                  norm=kwargs['norm'])),
                        ("svm", SVC(C=kwargs['C'],
                                    gamma=kwargs['gamma'],
                                    kernel=kwargs['kernel']))]
            pipe = Pipeline(objs)
            self.model = pipe.fit(X, y)
            print("Training complete")

        elif self.text_vectorization != 'tfidf' and fine_tune == True:
            print("Training and fine tunning the model")
            self.model = self.fit_and_optimize(X, y)

            print("Fine tunning metrics:")
            print("-"*10)
            print("CV Score using best parameter values:", self.model.best_score_)
            print("Best parameter values:")
            for param in self.model.best_params_.items():
                print(param)
            print("-"*10)

            print("Training and fine tunning complete")
        else:
            print("Error")

    def predict(self, X):
        return self.model.predict(X)

