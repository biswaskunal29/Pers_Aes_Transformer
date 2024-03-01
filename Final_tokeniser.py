from textblob import TextBlob

def tokenise(x):
    if type(x) is str:
        x_list = []
        x_list = TextBlob(x).words
        return x_list
    else:
        return x

# =============================================================================
# sample = "herbal care is . a good care."
# result = tokenise(sample)
# print(result)
# =============================================================================
