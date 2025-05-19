#1955年，卡普耶卡对4位数字进行了研究，发现一个规律：对任意各位数字不相同的4位数，使用各位数字能组成的最大数减去能组成的最小数，对得到的差重复这个操作，最终会得到6174这个数字，并且这个操作最多不会超过7次。
#编写程序，使用枚举法对这个猜想进行验证。

def kpyk(strnum):
    minnum = int("".join(sorted(str(strnum))))
    maxnum = int("".join(sorted(str(strnum), reverse=True)))
    result = maxnum - minnum
    return result



# strnum = input("请输入一个任意个位数字都不相同的四位数:")
# number = int(strnum)
# if number > 10000 or number < 1000:
#     print("您输入的不是四位数。")
# if len(strnum) != len(set(strnum)):
#     print("您输入的不是一个任意个位数字都不相同的四位数。")

numbers = []

for num in range(1000, 10000):  # 所有四位数
    strnum = str(num)
    if len(set(strnum)) == 4:  # 每位都不相同
        numbers.append(num)

# minnum = int("".join(sorted(strnum)))
# maxnum = int("".join(sorted(strnum, reverse=True)))

for number in numbers:
    result = kpyk(number)
    for i in range(1,8):
        if result != 6174:
            result = kpyk(result)
        else:
            break
    
    if result != 6174:
        print("error")
    else:
        print(number," ", result)
        print("true")


# result = kpyk(strnum)

# for i in range(1,7):
#     if result != 6174:
#         result = kpyk(result)
#         i = i + 1
#     else:
#         break

# if result != 6174:
#     print("error")
# else:
#     print(result)
#     print("true")






