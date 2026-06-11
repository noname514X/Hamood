with open('/Users/ncc1031a/Documents/VSCode/PythonProgramDesign/week14/sample.txt','r') as f:
    s = f.read(6)

print(s)  # 输出前6个字符
print('string length:', len(s))  # 输出字符串长度