'''编写程序，用户输入一个三位以上的整数，输出其百位以上的数字。
例如用户输入1234，则程序输出12（提示：使用整除运算）。
'''

number = input("请输入一个三位以上的整数：")
if len(number) < 3:
    print("输入的数字不符合要求")
    exit()
list1 = list(number)
list1.pop()
list1.pop()
outputnumber = "".join(list1)
print("输出为：",outputnumber)

