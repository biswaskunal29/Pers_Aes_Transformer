import re

#text = "hi email me @ thismail@gmail.com or something947@yahoo.com"

def rem_mail(x):
    if type(x) is str:
        x = re.sub(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)','',x)
        return x
    else:
        return x
    
#print(rem_mail(text))
