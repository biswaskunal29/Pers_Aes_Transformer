# RT stands for re-tweet
import re

def rem_RT(x):
    if type(x) is str:
        x = re.sub('RT','',x)
        return x
    else:
        return x

# =============================================================================
# sample = "sdasdda RT asdas"
# result = rem_RT(sample)
# print(result)
# =============================================================================
