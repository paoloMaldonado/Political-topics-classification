import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

class DataGenerator(Sequence):
    def __init__(self, kind, data, labels, n_classes, output_size, embedding_model, onehot_model=None, shuffle=True, batch_size=32):
        self.kind = kind
        self.df = data
        self.labels = labels
        self.n_classes = n_classes
        self.output_size = output_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        if kind == 'training':
          self.onehot_encoder = OneHotEncoder(sparse_output=False)
          self.onehot_encoder.fit(self.df.party.values.reshape(-1, 1))
        if kind == 'testing' or kind == 'validation':
          if onehot_model == None:
            raise Exception("No OneHotEncoder model detected, please make sure onehot_model has a valid trained model")
          else:
            self.onehot_encoder = onehot_model
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) # shuffle: each batch is given 32 random sentences from the dataset

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        # X is (batch_size, number of tokens, embedding size) -> (32, 60, 300)
        X = {'prev_phrase': np.empty((self.batch_size, *self.output_size)), 
             'phrase': np.empty((self.batch_size, *self.output_size)),
             'party': np.empty((self.batch_size, 1, 78))} # there are 78 political parties
        #X = {'phrase': np.empty((self.batch_size, *self.output_size))}
        y = np.empty((self.batch_size), dtype=int)

        # get the indices of the requested batch
        # if the size of the data is not a multiple of the batch size, the last
        # batch might be smaller
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        for i, data_index in enumerate(indexes):
            phrase      = self.df.iloc[data_index, 0] # 0 is the column 'phrase'
            prev_phrase = self.df.iloc[data_index, 1] # 1 is the column 'prev_phrase'
            party       = self.df.iloc[data_index, 2] # 2 is the column 'party'

            # Store sample
            X['prev_phrase'][i,] = self.__getWordMatrixPadded(prev_phrase, self.embedding_model)
            X['phrase'][i,]      = self.__getWordMatrixPadded(phrase, self.embedding_model)
            X['party'][i,]       = self.onehot_encoder.transform(np.array(party).reshape(-1, 1))
            
            # Store label
            y[i] = self.df.iloc[data_index, 3] # 3 is the column 'label'

        return X, to_categorical(y, num_classes=self.n_classes, dtype='int')
    
    def __getWordMatrixPadded(self, phrase, embedding_model):
        max_vocab, emb_dimension = self.output_size
        z = np.zeros(shape=(max_vocab, emb_dimension))
        # vectorize the sentence, if a phrase has no previous
        # one, then the previous sentence is a vector of zeros
        # of shape (max_vocab, emb_dimension) 
        try:
            word_vector = embedding_model[phrase]
        except ValueError:
            word_vector = np.zeros(shape=(max_vocab, emb_dimension))
        z[0:word_vector.shape[0], 0:word_vector.shape[1]] = word_vector[0:max_vocab, 0:emb_dimension]
        return z