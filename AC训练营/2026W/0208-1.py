
import sys
import re
import ast

def nthUglyNumber(n: int) -> int:
    i2=i3=i5=0
    ugly=[1]
    for _ in range(1,n):
        a=ugly[i2]*2
        b=ugly[i3]*3
        c=ugly[i5]*5
        nxt=min(a,b,c)
        ugly.append(nxt)
        if nxt==a:
            i2+=1
        if nxt==b:
            i3+=1
        if nxt==c:
            i5+=1
    return ugly[-1]

def main():
    data=sys.stdin.read()
    m=re.search(r"n\s*=\s*(-?\d+)",data)
    if m:
        n=int(m.group(1))
    else:
        try:
            val=ast.literal_eval(data.strip())
            if isinstance(val,dict) and 'n' in val:
                n=int(val['n'])
            elif isinstance(val,(list,tuple)):
                n=int(val[0])
            else:
                nums=re.findall(r"-?\d+",data)
                n=int(nums[0]) if nums else 0
        except Exception:
            nums=re.findall(r"-?\d+",data)
            n=int(nums[0]) if nums else 0
    print(nthUglyNumber(n))

if __name__=='__main__':
    main()

