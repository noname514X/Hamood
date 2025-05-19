def demo(*a):
    avg = sum(a) / len(a)
    g = [i for i in a if i > avg]
    return (avg,) + tuple(g)

print(demo(1,2,3,4))

def demo2(a,b,c):
    print(a,b,c)

demo2(*(1,2,3))
demo2(1,*(2,3))
demo2(1,*(2,),3)
demo2(*(1,2),c=3)