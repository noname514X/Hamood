'''
实现一个程序来存放你的日程安排。如果要添加的时间内不会导致三重预订时，则可以存储这个新的日程安排。
当三个日程安排有一些时间上的交叉时（例如三个日程安排都在同一时间内），就会产生三重预订。
事件能够用一对整数 startTime 和 endTime 表示，在一个半开区间的时间［startTime, endTime）上预定。实数 ×的范围为 startTime <= x< endTime。
实现 MyCalendarTwo类：
• MyCalendarTwo（）初始化日历对象。
• boolean book（int startTime, int endTime）如果可以将日程安排成功添加到日历中而不会导致三重预订，返回true。否则，返回 false 并且不要将该日程安排添加到日历中。
'''

class MyCalendarTwo:

    def __init__(self):
        self.bookings = []
        self.double_bookings = []

    def book(self, startTime: int, endTime: int) -> bool:
        for start, end in self.double_bookings:
            if start < endTime and startTime < end:
                return False
        for start, end in self.bookings:
            if start < endTime and startTime < end:
                self.double_bookings.append((max(start, startTime), min(end, endTime)))
        self.bookings.append((startTime, endTime))
        return True

if __name__ == "__main__":
    my_calendar_two = MyCalendarTwo()
    print(my_calendar_two.book(10, 20))  
    print(my_calendar_two.book(15, 25))  
    print(my_calendar_two.book(20, 30))  
    print(my_calendar_two.book(5, 15))   
    print(my_calendar_two.book(25, 35))
