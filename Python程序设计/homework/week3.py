# 使用列表实现筛选法求素数
# 编写程序，输入一个大于2的自然数，然后输出小于该数字的所有素数组成的列表。所谓素数，是指除了1和自身之外没有其他因数的自然数，最小的素数是2，后面依次是3 5 7 11 13
# 提示：算法考虑埃氏筛或者欧拉筛。

n = int(input())
primes = []
candidates = list(range(2, n))

while candidates:
    p = candidates[0]
    primes.append(p)
    candidates = [num for num in candidates if num % p != 0]
    if p * p > n:
        primes += candidates
        break

print(primes)