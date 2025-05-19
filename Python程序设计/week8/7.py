from string import ascii_letters, ascii_lowercase, ascii_uppercase

def kaisaEncrypt(text,k):
    lower = ascii_lowercase[k:] + ascii_lowercase[:k]
    upper = ascii_uppercase[k:] + ascii_uppercase[:k]
    table = ' '.maketrans(ascii_letters, lower+upper)
    return text.translate(table)

s = 'ZWJ is hamood, and he likes CCB'
print(kaisaEncrypt(s, 3))



def check(text):
    mostCommonWords = ('the', 'is', 'to', 'not', 'have', 'than', 'for')
    return sum(1 for word in mostCommonWords if word in text.lower()) >= 2


tt = kaisaEncrypt(s, 3)
for i in range(1,26):
    t = kaisaEncrypt(tt, -i)
    if check(t):
        print(i,t,sep=":")
        break