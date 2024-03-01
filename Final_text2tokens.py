from Final_full_preprocessor import full_preprocess_text_to_token_list as text2tokens


text = """CapiTal LEttErs
he's who'd
mail sdasd@sds.com was here
url https://www.google.com/search?q=python+make was here and https://twitter.com/thenewsminute/photo here
this is a small rt and a CAP RT by musk
ą/ę/ś/ć ° Ö ääliö A \u00c0 Montréal, über, 12.89, Mère, Françoise, noël, 889 ø ł
@mcdonald get me #cheaper meals
this i's baddly '.'''][\=-] punctuated $%^
this   has      a lot   of spaces
<html><h2>This is html tagged</h2></html>
The striped bats were hanging on their feet and ate best fishes that dived
tokenise this


@tbiz: I KNEW @pfinette was he'd behind this - verified by @thunder - looks great! #mozillapersona """

result = text2tokens(text)
print(result)