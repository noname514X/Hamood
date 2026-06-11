'''
这里有n个航班，它们分别从 1 到n进行编号。
有一份航班预订表 bookings，表中第i条预订记录 bookings ［i］ = ［firsti,lasti,seatsi］意味着在从 firsti到 lasti
请你返回一个长度为 n的效组 answer，里面的元素是每个航班预定的座位总效。
'''
def corpFlightBookings(bookings, n):
    answer = [0] * n
    for first, last, seats in bookings:
        answer[first - 1] += seats
        if last < n:
            answer[last] -= seats
    
    for i in range(1, n):
        answer[i] += answer[i - 1]
    
    return answer

if __name__ == "__main__":
    bookings1 = [[1, 2, 1], [2, 3, 2], [2, 5, 3]]
    n1 = 5
    print(corpFlightBookings(bookings1, n1)) 
    bookings2 = [[1, 2, 3], [2, 3, 4], [1, 5, 2]]
    n2 = 5
    print(corpFlightBookings(bookings2, n2)) 
