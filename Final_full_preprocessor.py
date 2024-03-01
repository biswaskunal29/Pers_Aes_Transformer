# Dated 18-sep-2020

from Final_cont2exp import cont2exp as expander
from Final_remove_emails import rem_mail as remove_mails
from Final_remove_url import rem_url as remove_urls
from Final_remove_RT import rem_RT as remove_RTs
from Final_remove_accent import rem_acc as remove_accent
from Final_remove_at_hash import rem_at_hash as remove_at_hashtags
from Final_remove_punc_spl_char import rem_punc as remove_punct_spl_char
from Final_remove_whitespace import rem_wspace as remove_wspace
from Final_remove_htmltags import rem_html as remove_html_tags
from Final_lemmatiser import make_baseform as to_baseform
from Final_tokeniser import tokenise
from Final_us_to_uk_converter import replace_all as us2uk

def full_preprocess_text_to_token_list(x):
    if type(x) is str:
        #case sensitive steps first
        p = remove_RTs(x)
        
        #lowercase and rest
        a = p.lower()           #lowercase
        b = expander(a)
        c = remove_mails(b)
        d = remove_urls(c)
        e = remove_accent(d)
        f = remove_at_hashtags(e)
        g = remove_html_tags(f)
        h = remove_punct_spl_char(g)
        i = remove_wspace(h)
        j = to_baseform(i)
        k = us2uk(j)
#        Now tokenising
        l = []
        l = tokenise(k)
        
        return l
    else:
        return x






# =============================================================================
# text = """CapiTal LEttErs
# he's who'd
# mail sdasd@sds.com was here
# url https://www.google.com/search?q=python+make was here and https://twitter.com/thenewsminute/photo here
# this is a small rt and a CAP RT by musk
# ą/ę/ś/ć ° Ö ääliö A \u00c0 Montréal, über, 12.89, Mère, Françoise, noël, 889 ø ł
# @mcdonald get me #cheaper meals
# this i's baddly '.'''][\=-] punctuated $%^
# this   has      a lot   of spaces
# <html><h2>This is html tagged</h2></html>
# The striped bats were hanging on their feet and ate best fishes that dived
# tokenise this
# color stabilize
# uk	american
# 
# @tbiz: I KNEW @pfinette was he'd behind this - verified by @thunder - looks great! #mozillapersona """
# 
# result = full_preprocess_text_to_token_list(text)
# print(result)
# =============================================================================


