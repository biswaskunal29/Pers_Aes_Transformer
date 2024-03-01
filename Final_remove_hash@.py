import re

def rem_at_hash(x):
    if type(x) is str:
        x = re.sub(r"#(\w+)", ' ', x, flags=re.MULTILINE)
        x = re.sub(r"@(\w+)", ' ', x, flags=re.MULTILINE)           
        return x
    else:
        return x
    
# =============================================================================
# sample = "here is a #GoFlyAway for u @littlebird"
# result = rem_at_hash(sample)
# print(result)
# =============================================================================
