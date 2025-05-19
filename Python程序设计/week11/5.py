def rate(origin, userInput):
    if not (isinstance(origin, str) and isinstance(userInput, str)):
        print('The two parameters must be strings.')
        return
    if len(origin) < len(userInput):
        print('Sorry. I suppose the second parameter string is shorter.')
        return
    right = sum(map(lambda oc, uc:oc==uc, origin, userInput))
    return right / len(origin)

print(rate('hello', 'hell'))
print(rate('hello', 'hello'))