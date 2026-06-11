'''
实现一个 MyCalendar 类来存放你的日程安排。如果要添加的日程安排不会造成 重复预订，则可以存储这个新的日程安排。
当两个日程安排有一些时间上的交叉时（例如两个日程安排都在同一时间内），就会产生 重复预订。
日程可以用一对整数 startTime 和 endTime 表示，这里的时间是半开区间，即 ［startTime,endTime），实数 x的范围为，startTime <= x < endTime。
实现 MyCalendar类：
• MyCalendar（）初始化日历对象。
• boolean book（int startTime, int endTime）如果可以将日程安排成功添加到日历中而不会导致重复预订，返回 true。否则，返回 false 并且不要将该日程安排添加到日历中。
'''
class MyCalendar:

    def __init__(self):
        self.bookings = []

    def book(self, startTime: int, endTime: int) -> bool:
        for start, end in self.bookings:
            if start < endTime and startTime < end:
                return False
        self.bookings.append((startTime, endTime))
        return True

if __name__ == "__main__":
    my_calendar = MyCalendar()
    print(my_calendar.book(10, 20))  
    print(my_calendar.book(15, 25))  
    print(my_calendar.book(20, 30))  
    print(my_calendar.book(5, 15))   
    print(my_calendar.book(25, 35))  
