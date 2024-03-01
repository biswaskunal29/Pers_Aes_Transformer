import re

def rem_punc(x):
    if type(x) is str:
        x = re.sub('[^A-Z a-z \n \t 0-9-]+','',x)
        return x
    else:
        return x
    
# =============================================================================
# sample = "yen;na ras45f4=-kl.,mk\\\""
# result = rem_punc(sample)
# print(result)
# =============================================================================
