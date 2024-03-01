import re

def rem_wspace(x):
    if type(x) is str:
        x = re.sub(" +", " ",x)
        x = x.lstrip()
        x = x.rstrip()
#        x = ' '.join(x.split())       
        return x
    else:
        return x
    
# =============================================================================
# sample = """ CapiTal LEttErs
# he's who'd
# mail sdasd@sds.com was here"""
# result = rem_wspace(sample)
# print(result)
# =============================================================================
