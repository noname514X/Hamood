import sys
import re
import ast

def merge(nums1,m,nums2,n):
    i=m-1
    j=n-1
    k=m+n-1
    while j>=0:
        if i>=0 and nums1[i]>nums2[j]:
            nums1[k]=nums1[i]
            i-=1
        else:
            nums1[k]=nums2[j]
            j-=1
        k-=1


def main():
    data=sys.stdin.read()
    arrs=re.findall(r"\[[\s\S]*?\]",data)
    nums1=[]
    nums2=[]
    if len(arrs)>=1:
        try:
            nums1=ast.literal_eval(arrs[0])
        except Exception:
            nums1=[]
    if len(arrs)>=2:
        try:
            nums2=ast.literal_eval(arrs[1])
        except Exception:
            nums2=[]
    m_match=re.search(r"\bm\s*=\s*(\d+)",data)
    n_match=re.search(r"\bn\s*=\s*(\d+)",data)
    m=len(nums1) if not m_match else int(m_match.group(1))
    n=len(nums2) if not n_match else int(n_match.group(1))
    merge(nums1,m,nums2,n)
    print(nums1)

if __name__=='__main__':
    main()
