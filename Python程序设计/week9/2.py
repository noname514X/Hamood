import re

a = 'hamoodccb is not ccb'
b = '^hamood'
print(re.findall(b, a))

c = r'^[a-zA-Z_][a-zA-Z0-9_]?'
d = '1a'
print(re.findall(c, d))
e = 'a1'
print(re.findall(c, e))


print(re.findall(b,a))