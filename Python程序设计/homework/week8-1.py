#一般地，可以作为密码字符的主要有数字、小写字母、大写字母和几个标点符号。
#密码安全强度主要和字符串的复杂程度有关系，字符串中包含的字符种类越多，认为其安全强度越高。
#按照这个标准，可以把安全强度分为强密码、中高、中低、弱密码。
# 其中强密码表示字符串中同时含有数字、小写字母、大写字母、标点符号这4类字符，而弱密码表示字符串中仅包含4类字符中的一种。
#编写程序，输入一个字符串，输出该字符串作为密码时的安全强度。


import string

passwords = input('Password:')



def isLower(passwords):
    for password in passwords:
        if password in string.ascii_lowercase:
            return True
    return False

def isUpper(passwords):
    for password in passwords:
        if password in string.ascii_uppercase:
            return True
    return False

def isNumber(passwords):
    for password in passwords:
        if password in string.digits:
            return True
    return False

def isPunctuation(passwords):
    for password in passwords:
        if password in string.punctuation:
            return True
    return False
    
def password_strength(passwords):
    types = 0
    if isLower(passwords):
        types += 1
    if isUpper(passwords):
        types += 1
    if isNumber(passwords):
        types += 1
    if isPunctuation(passwords):
        types += 1
        
    if types == 4:
        print("Very Strong,强密码。")
    elif types == 3:
        print("Strong,中高强度密码。")
    elif types == 2:
        print("Not Strong,中低强度密码。")
    elif types == 1:
        print("Weak,弱密码。")
    else:
        print("Invalid password,密码无效!")

password_strength(passwords)