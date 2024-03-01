from textblob import TextBlob

def spell_correct(x):
    if type(x) is str:
        x = TextBlob(sample).correct()
        return x
    else:
        return x


sample = "tanks forr waching regin off border carr"
result = spell_correct(sample)
print(result)
