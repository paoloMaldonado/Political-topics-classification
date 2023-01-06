import numpy as np

def identity_tokenizer(text):
    return text

# Returns the TFIDF weights of each token-sentence pair in terms of the order of the sentence
def getTFIDFtokenSentenceWeight(sentence, vocabulary, array_sentence):
    sentenceTFIDFweights = []
    for word in sentence:
        try:
            word_index = vocabulary[word]
        except KeyError: # for test set if there not exists that word in the current vocabulary (no need for training a new one)
            sentenceTFIDFweights.append(1.0)
            continue
        weight = array_sentence[word_index]
        sentenceTFIDFweights.append(weight)
    return np.asarray(sentenceTFIDFweights)