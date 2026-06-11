'''
给你一个日志数组 1ogs。每条日志都是以空格分隔的字串，其第一个字为字母与数字混合的 标识符。
有两种不同类型的日志：
•字母日志：除标识符之外，所有字均由小写字母组成
•数字日志：除标识符之外，所有字均由数字组成
请按下述规则将日志重新排序：
• 所有 字母日志 都排在 数字日志之前。
•字母日志 在内容不同时，忽略标识符后，按内容字母顺序排序；在内容相同时，按标识符排序。
• 数字日志 应该保留原来的相对顺序。
返回日志的最终顺序。
'''
def reorderLogs(logs):
    def is_digit_log(log):
        return log.split()[1].isdigit()

    letter_logs = []
    digit_logs = []

    for log in logs:
        if is_digit_log(log):
            digit_logs.append(log)
        else:
            letter_logs.append(log)


    letter_logs.sort(key=lambda x: (x.split()[1:], x.split()[0]))

    return letter_logs + digit_logs

if __name__ == "__main__":
    logs = [
        "dig1 8 1 5 1",
        "let1 art can",
        "dig2 3 6",
        "let2 own kit dig",
        "let3 art zero"
    ]
    print(reorderLogs(logs))
