def rem_acc(x):
    if type(x) is str:
        x = x.lower()
        return x
    else:
        return x

sample = u'hhdsSHDJss jsja HDJDJShshd'
result = rem_acc(sample)
print(result)
