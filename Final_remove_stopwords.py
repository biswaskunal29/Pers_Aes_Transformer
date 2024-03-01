#import spacy
from spacy.lang.en.stop_words import STOP_WORDS

def rem_stopwords(x):
    if type(x) is str:
        x = " ".join([t for t in x.split() if t not in STOP_WORDS])
        return x
    else:
        return x


sample = "this a stop word from 1990 removal code. how do you do that?"
result = rem_stopwords(sample)
print(result)


