from bs4 import BeautifulSoup

def rem_html(x):
    if type(x) is str:
        x = BeautifulSoup(x, 'lxml').get_text()
        return x
    else:
        return x

# =============================================================================
# sample = "<html><h2>This is inside tags</h2></html>"
# result = rem_html(sample)
# print(result)
# =============================================================================
