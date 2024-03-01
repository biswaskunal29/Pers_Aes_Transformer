import unidecode

def rem_acc(x):
    if type(x) is str:
        x = unidecode.unidecode(x)
        return x
    else:
        return x

# =============================================================================
# sample = u'ą/ę/ś/ć ° Ö ääliö A \u00c0 \u0394 \u038E Montréal, über, 12.89, Mère, Françoise, noël, 889 ø ł'
# result = rem_acc(sample)
# print(result)
# 
# =============================================================================
