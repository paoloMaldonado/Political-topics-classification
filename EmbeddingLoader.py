from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors, load_facebook_model

class EmbeddingLoader:
    def __init__(self, embedding_name, embedding_path, for_update=False):
        self.embedding_name = embedding_name
        self.embedding_path = embedding_path
        self.embedding_object = None
        self.for_update = for_update
    def load_embedding_model(self):
        if self.embedding_name == "word2vec":
            self.embedding_object = KeyedVectors.load_word2vec_format(self.embedding_path, binary=True)
        elif self.embedding_name == "fasttext":
            if self.for_update == False: # load just keyedVectors 
                print("Loading keyedVectors fasttext model")
                self.embedding_object = load_facebook_vectors(self.embedding_path)
            else:
                print("Loading fasttext model for finetuning")
                self.embedding_object = load_facebook_model(self.embedding_path)
        else:
            print("Model not recognized")
        return self