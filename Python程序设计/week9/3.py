import re

exampleString = 'nicknack paddy whack give a dog a bone nicknack paddy whack give a dog a bone'
pattern = re.compile(r'\b(?i)n\w+\b')
index = 0
while True:
    matchResult = pattern.search(exampleString, index)
    if not matchResult:
        break
    print(matchResult).group(0), ':', matchResult.span(0)
    index = matchResult.end(0)
