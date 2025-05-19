x = int(input("x = "))

if (x < 0):
    print("y = ", 0)
elif (0 <= x < 5):
    print("y =", x)
elif (5 <= x < 10):
    print("y =", 3*x - 5)
elif (10 <= x < 20):
    print("y =", 0.5*x - 2)
else:
    print("y =", 0)