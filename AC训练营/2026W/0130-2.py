import sys,re

def restore(s):
    n=len(s)
    res=[]
    if n<4 or n>12:
        return res
    def valid(seg):
        if not seg:
            return False
        if seg[0]=='0' and len(seg)>1:
            return False
        return int(seg)<=255
    for i in range(1,min(4,n-2)):
        a=s[:i]
        if not valid(a):
            continue
        for j in range(i+1, i+min(4,n-i-1)):
            b=s[i:j]
            if not valid(b):
                continue
            for k in range(j+1, j+min(4,n-j)):
                c=s[j:k]
                d=s[k:]
                if valid(c) and valid(d):
                    res.append('.'.join([a,b,c,d]))
    return res


def main():
    data=sys.stdin.read()
    m=re.search(r'"(\d+)"',data)
    if not m:
        m=re.search(r"'(\d+)'",data)
    if m:
        s=m.group(1)
    else:
        m=re.search(r'\d+',data)
        s=m.group(0) if m else ''
    print(restore(s))

if __name__=='__main__':
    main()
