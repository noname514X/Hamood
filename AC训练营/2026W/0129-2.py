import sys
import re
data=sys.stdin.read()
words=re.findall(r'"([a-z]+)"',data.lower())+re.findall(r"'([a-z]+)'",data.lower())
if len(words)<2:
    tokens=re.findall(r'[a-z]+',data.lower())
    words=words+tokens
if len(words)>=2:
    w1,w2=words[0],words[1]
else:
    w1,w2="",""
if len(w2)>len(w1):
    w1,w2=w2,w1
m=len(w2)
prev=list(range(m+1))
for i,ch in enumerate(w1,1):
    cur=[i]+[0]*m
    for j in range(1,m+1):
        if ch==w2[j-1]:
            cur[j]=prev[j-1]
        else:
            a=prev[j]
            b=cur[j-1]
            c=prev[j-1]
            cur[j]=1+ (a if a<=b and a<=c else b if b<=a and b<=c else c)
    prev=cur
print(prev[m])
