import sys,re,ast

def rob(nums):
    a=0
    b=0
    for x in nums:
        a,b=b,max(b,a+x)
    return b

def main():
    data=sys.stdin.read()
    arrs=re.findall(r"\[[\s\S]*?\]",data)
    nums=[]
    if arrs:
        try:
            nums=ast.literal_eval(arrs[0])
        except Exception:
            nums=[]
    print(rob(nums))

if __name__=='__main__':
    main()
