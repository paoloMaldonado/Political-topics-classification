import string
exclude = set(string.punctuation)

def removePunctuation(token_list):
    return [token for token in token_list if token not in exclude]

def replace_numbers(token_list):
    s = []
    for token in token_list:
        if is_number(token):
            s.append("DIGITO")
        else:
            s.append(token)
    return s

def is_number(s):
    try:
        float(s)    
        return True
    except ValueError:
        return False