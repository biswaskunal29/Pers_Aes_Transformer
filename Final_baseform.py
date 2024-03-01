#This will take a huge Time

import spacy

def make_baseform(x):
#    nlp=spacy.load('en_core_web_sm')
    if type(x) is str:
        nlp=spacy.load('en_core_web_sm')
        x_list = []
        doc = nlp(x)
        for token in doc:
            lemma = str(token.lemma_)
            if lemma == '-PRON-' or lemma == 'be':
                lemma = token.text
            x_list.append(lemma)
            x = " ".join(x_list)
        return x
    else:
        return x

sample = "The striped bats were hanging on their feet and ate best fishes dived"
result = make_baseform(sample)
print(result)
