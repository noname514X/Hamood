
sb = 'hamood'

def rev1(s):
    return ' '.join(reversed(s.split()))
def rev2(s):
    t = s.split()
    t.reverse()
    return ' '.join(t)
def rev3(s):
    t = ''.join(reversed(s)).split()
    t = map(lambda x: ''.join(reversed(x)), t)
    return ' '.join(t)

print(rev1(sb))
print(rev2(sb))
print(rev3(sb))