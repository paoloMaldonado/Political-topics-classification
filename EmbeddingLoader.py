from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors

class EmbeddingLoader:
    def __init__(self, embedding_name, embedding_path):
        self.embedding_name = embedding_name
        self.embedding_path = embedding_path
        self.embedding_object = None
    def load_embedding_model(self):
        if self.embedding_name == "word2vec":
            self.embedding_object = KeyedVectors.load_word2vec_format(self.embedding_path+"/word2vec/SBW-vectors-300-min5-cbow.bin", binary=True)
        elif self.embedding_name == "fasttext":
            self.embedding_object = load_facebook_vectors(self.embedding_path+"/fasttext/fasttext_model_100.bin")
        else:
            print("Model not recognized")
        return self