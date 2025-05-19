
class Root:
    __total = 0

    def __init__(self, v):
        self.__value = v
        Root.__total += 1

    def show(self):
        print('self.__value:', self.__value)
        print('Root.__total:', Root.__total)

    @classmethod
    def classShowTotal(cls):
        print(cls.__total)

    @staticmethod
    def staticShowTotal():
        print


