from utils.utils_tfidf import getTFIDFtokenSentenceWeight
from collections import Counter
import numpy as np

# def checkOOVwords(vocabulary, embedding, verbose=0):
#     oov_counter = 0
#     oov_words = []
#     if verbose == 1:
#         print("OOV words:")
#         print("-"*5)
#     for word in vocabulary:
#         try:
#             embedding.get_vector(word)
#         except KeyError:
#             if verbose == 1:
#                 print(word)
#             oov_words.append(word)
#             oov_counter+=1
#     #print("Number of OOV words:", oov_counter)
#     return oov_words

def checkOOVwords(phrases, embedding, verbose=0):
    # Creating the vocabulary
    all_words = []
    for sentence in phrases:
        all_words.extend(sentence)
    counter = Counter(all_words)
    vocabulary = list(counter.keys())

    # Create list of oov words
    oov_words = []
    for w in vocabulary:
        if w not in embedding.key_to_index:
            oov_words.append(w)

    print('Number of OOV words {}'.format(len(oov_words)))
    print('They represent {0:.2f}% of all the vocabulary of words'.format(len(oov_words)/len(vocabulary)*100))

    return oov_words

def createRandomOOV(oov_words, embedding_shape=(300,)):
    oov_to_vector = {}
    for word in oov_words:
        if word not in oov_to_vector.keys():
            oov_to_vector[word] = np.random.uniform(low=-1.0, high=1.0, size=embedding_shape)
    return oov_to_vector

def wcbow(sentences, vocabulary, array_sentences, embedding, oov_to_vector, normalize=False):
    # each row represents the dimension of the sentence emebedding, and each column is a sentence
    sentence_matrix = np.ones(shape=(len(sentences), embedding[0].shape[0]))
    # for every sentence
    for i, sentence in enumerate(sentences):
        # get the tfidf weights of the ith sentence
        tfidf_weights = getTFIDFtokenSentenceWeight(sentence, vocabulary, array_sentences[i])
        # for every token in the sentence
        word_vector_sum = np.zeros(shape=embedding[0].shape) #set to embedding size
        for k, word in enumerate(sentence):
            try:
                word_vector = embedding.get_vector(word, norm=normalize)
            except KeyError:
                try:
                    word_vector = oov_to_vector[word]
                except KeyError: # if there is no word in oov list just generates a random vector (this is for test set)
                    word_vector = np.random.uniform(low=-1.0, high=1.0, size=embedding[0].shape)
            # get the tfidf weights of the kth word
            word_vector_sum += tfidf_weights[k]*word_vector
        sentence_vector = word_vector_sum/np.sum(tfidf_weights)
        sentence_matrix[i] = sentence_vector
    return sentence_matrix

def avgcbow(sentences, embedding, oov_to_vector, normalize=False):
    # each row represents the dimension of the sentence emebedding, and each column is a sentence
    sentence_matrix = np.ones(shape=(len(sentences), embedding[0].shape[0]))
    # for every sentence
    for i, sentence in enumerate(sentences):
        # for every token in the sentence
        word_vector_sum = np.zeros(shape=embedding[0].shape) #set to embedding size
        for word in sentence:
            try:
                word_vector = embedding.get_vector(word, norm=normalize)
            except KeyError:
                try:
                    word_vector = oov_to_vector[word]
                except KeyError: # if there is no word in oov list just generates a random vector (this is for test set)
                    word_vector = np.random.uniform(low=-1.0, high=1.0, size=embedding[0].shape)
            word_vector_sum += word_vector
        sentence_vector = word_vector_sum/len(sentence)
        sentence_matrix[i] = sentence_vector
    return sentence_matrix