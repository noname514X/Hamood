class Car:
    price = 100000
    def __init__(self,c):
        self.color = c

car1 = Car('Red')
car2 = Car('Blue')
print(car1.color)
print(car2.color)
print(car1.price)
print(car2.price)
print(Car.price)

