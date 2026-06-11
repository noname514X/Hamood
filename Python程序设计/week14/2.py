import os

fns = (fn for fn in os.listdir() if fn.endswith('.txt'))
for fn in fns:
    try:
        with open(fn, encoding = 'utf-8') as f:
            f.read()
    except:
        with open(fn) as fp1:
            with open('t.txt','w',encoding='utf-8') as fp2:
                fp2.write(fp1.read())
        os.remove(fn)
        os.rename('t.txt', fn)