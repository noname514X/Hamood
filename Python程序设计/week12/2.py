class A:
    def __init__(self, value1=0, value2=0):
        self.value1 = value1
        self.__value2 = value2
    def setValue(self, value1, value2):
        self.vallule1 = value1
        self.__value2 = value2
    def show(self):
        print(self.value1)
        print(self.__value2)

a = A(1,2)
print(a.value1)
print(a._A__value2) 