text = '''
东边来个小朋友叫小松，手里拿着一捆葱。
西边来个小朋友叫小丛，手里拿着小闹钟。
小松手里葱捆得松，掉在地上一些葱。
小丛忙放闹钟去拾葱，帮助小松捆紧葱.
小松夸小丛像雷锋，小从说小松爱劳动。
'''

for index, ch in enumerate(text):
    if index == text.index(ch):
        print((index, ch), end = '')