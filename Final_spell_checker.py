from textblob import TextBlob




sample = "tanks forr waching regin off border carr"
x = TextBlob(sample).correct()
print(x)
