import sys
import re
import ast

data=sys.stdin.read().strip()
if '=' in data:
    n_match=re.search(r"numerator\s*=\s*(-?\d+)",data)
    d_match=re.search(r"denominator\s*=\s*(-?\d+)",data)
    if n_match and d_match:
        numerator=int(n_match.group(1))
        denominator=int(d_match.group(1))
    else:
        nums=re.findall(r"-?\d+",data)
        numerator=int(nums[0])
        denominator=int(nums[1])
else:
    try:
        vals=ast.literal_eval(data)
        if isinstance(vals,dict):
            numerator=vals.get('numerator')
            denominator=vals.get('denominator')
        elif isinstance(vals,(list,tuple)):
            numerator=vals[0]
            denominator=vals[1]
        else:
            parts=re.findall(r"-?\d+",data)
            numerator=int(parts[0])
            denominator=int(parts[1])
    except Exception:
        parts=re.findall(r"-?\d+",data)
        numerator=int(parts[0])
        denominator=int(parts[1])

if numerator==0:
    print("0")
    sys.exit(0)
sign = '-' if numerator*denominator<0 else ''
num=abs(numerator)
den=abs(denominator)
integer=num//den
rem=num%den
if rem==0:
    print(sign+str(integer))
    sys.exit(0)
digits=[]
pos={}
while rem and rem not in pos:
    pos[rem]=len(digits)
    rem*=10
    digits.append(str(rem//den))
    rem=rem%den
if rem==0:
    print(sign+str(integer)+'.'+''.join(digits))
else:
    idx=pos[rem]
    print(sign+str(integer)+'.'+''.join(digits[:idx])+'('+''.join(digits[idx:])+')')
