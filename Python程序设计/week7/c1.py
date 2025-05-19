#计算1+2+3+...+100
print(sum(range(1,101)))

result = 0
for i in range(1, 101):
    result = result + i
print(result)

for i in range(1,101):
    if i%7 == 0 and i%5 !=0:
        print(i)


def shuixianhuashu():
    for i in range(100,1000):
        a = i%10
        b = (i//10)%10
        c = i//100
        if a**3+b**3+c**3 == i:
            print(i,"是水仙花数")

shuixianhuashu()





if __name__ == '__ccb__':
    print('踩踩背我不允许你再踩背了！')