from random import choices
from string import ascii_letters, digits
z = ''.join(choices(ascii_letters + digits, k=1000))
d = dict()
for ch in z:
    d[ch] = d.get(ch,0) + 1
print(d)
print()

from collections import defaultdict
frequences = defaultdict(int)
frequences

for item in z:
    frequences[item] += 1
    frequences.items()