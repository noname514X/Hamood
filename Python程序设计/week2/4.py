list1 = [1,2,3,4,5]
list2 = list1 + [6,7,8,9,10]
list3 = list("hello world")

print(list1)
print(list2)
print(list3)

list3 = list3 + ['!']
print(list3)



import random
random.shuffle(list2)
print(list2)
list2.sort()
print(list2)