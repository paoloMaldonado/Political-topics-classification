import string
exclude = set(string.punctuation)

def removePunctuation(token_list):
    return [token for token in token_list if token not in exclude]

def is_number(s):
    try:
        float(s)    
        return True
    except ValueError:
        return False