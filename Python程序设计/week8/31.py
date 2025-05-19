text = '''
东边来个小朋友叫小哈姆，手里拿着一捆葱。
西边来个小朋友叫踩踩背，手里拿着小闹钟。
小哈姆手里葱捆得松，掉在地上一些葱。
踩踩背忙放闹钟去拾葱，帮助小哈姆捆紧葱.
小哈姆夸踩踩背像雷锋，踩踩背说爱劳动。
'''

for index, ch in enumerate(text):
    if index == text.index(ch):
        print((index, ch), end = '')