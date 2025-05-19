def demo(newitem, old_list=[]):
    old_list.append(newitem)
    return old_list

def demo2(newitem, old_list=None):
    if old_list is None:
        old_list = []
    new_list = old_list[:]
    new_list.append(newitem)
    return new_list

print(demo('5',[1,2,3,4]))
print(demo('aaa', ['a','b']))
print(demo('a'))
print(demo('b'))
print()
print(demo2('5',[1,2,3,4]))
print(demo2('aaa', ['a','b']))
print(demo2('a'))
print(demo2('b'))