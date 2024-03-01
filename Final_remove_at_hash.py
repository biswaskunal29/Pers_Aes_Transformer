import re

def rem_at_hash(x):
    if type(x) is str:
        x = re.sub(r"#(\w+)", ' ', x, flags=re.MULTILINE)
        x = re.sub(r"@(\w+)", ' ', x, flags=re.MULTILINE)           
        return x
    else:
        return x
    
# =============================================================================
# sample = "@ChronicleBooks: Chronicle Books at @Comic_Con—what you need to know  #SDCC2016 "
# result = rem_at_hash(sample)
# print(result)
# =============================================================================
