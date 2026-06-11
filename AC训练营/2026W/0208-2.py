
import sys
import re
import ast

def isUgly(n: int) -> bool:
    if n<=0:
        return False
    for p in (2,3,5):
        while n%p==0:
            n//=p
    return n==1

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
    print("true" if isUgly(n) else "false")

if __name__=='__main__':
    main()

