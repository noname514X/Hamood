import time
date = time.localtime()
year,month,day = date[:3]
print(year,month,day)

day_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

if year%400 or (year%4 == 0 and year%100 != 0):
    day_month[1] = 29

if month == 1:
    print(day)
else:
    for i in range(1,month):
        day += day_month[i]
    print(day)