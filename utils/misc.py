import string
exclude = set(string.punctuation)

def removePunctuation(token_list):
    return [token for token in token_list if token not in exclude and token != '[no_prev]']

def replaceNumbers(token_list):
    s = []
    for token in token_list:
        if isNumber(token):
            s.append("DIGITO")
        else:
            s.append(token)
    return s

def isNumber(s):
    try:
        float(s)    
        return True
    except ValueError:
        return False

def joinPhrases(prev, current):
    return prev + ' ' + current