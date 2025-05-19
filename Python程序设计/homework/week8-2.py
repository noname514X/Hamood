#一般来说，在一封正常邮件中，是不会出现太多类似于【、】、*、—、/这样的符号的。
# 如果一封邮件中包含的类似字符数量超过一定的比例，可以直接认为是垃圾邮件，而不需要朴素贝叶斯算法或者支持向量机等复杂的算法，可以大幅度提高分类速度。
# 编写程序，对给定的邮件内容进行分类，提示“垃圾邮件”或“正常邮件”。
import string

mails = "jdfaldjalsdjakjld[dp[[][][],..,/.....././././././.、、、、、、、、、、、、、、、、、w、wer、、、、、、、、、、、dsjflsdquoiwerueowir、、、、fscsvs、、、、、、、rewerdsdf、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、.,,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.-]-/;,,;';'dlas'dld'ald'sdls'adaspdad]adapd]asd]as[d.d;';'/;'.,ll'l'l]]oa],,asldsa.d]asd]///d.asd.]d/sd/sd;/./.,;'l';;l'./,/./.>?l]]p[]p][p][p';'./.]"
mails2 = '一二三四五，六七八九十。'
ChinesePunctuation = {'【','，','】','*','—','/','、'}

def check(mails):
    count = 0
    for mail in mails:
        if mail in string.punctuation or mail in ChinesePunctuation:
            count = count + 1
    
    if count > 50:
        print("垃圾邮件")
    else:
        print("正常邮件")



check(mails)
check(mails2)
check(input('请输入你的邮件以判断是否为垃圾邮件：'))