words = ('非法','无效','错误','失败','异常','bug','问题')

text = "这段话里含有非法内容"
for word in words:
    if word in text:
        text = text.replace(word, '***')
print(text)


