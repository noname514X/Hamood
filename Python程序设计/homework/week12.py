'''
请按要求自定义类并不断优化：
1. 设计一个学生Student类,类中属性包括id学号、name姓名、gender性别、age年龄；
具体设计要求如下：
（1）设计Student的构造函数Student(id, name, gender, age)（通过学号、姓名、性别、年龄构造，其中id,name,gender为字符串，age为整型）
（2）设计Student的方法getAge()返回该学生的年龄
（3）设计Student的方法setAge(newage)更改该学生的年龄
（4）定义Student的方法showInfo()显示学生的基本信息，返回格式化的字符串, 具体格式如id:3, name:sam, gender:male, age:20
2. 请构造如下两个实例并显示信息
（1）‘202201’, ’张三’, ’male’, 18
（2）‘202202’, ’李四’, ’female’, 20
3. 修改张三的年龄为19岁，重新显示信息
4. 设置一个类变量，统计现有学生人数
5. 新增一个学生，’202203’, ’王五’, ’female’, 17
6. 打印现有学生人数
程序保存为ex3_student.py。
'''
class Student:
    count = 0
    def __init__(self, id, name, gender, age):
        self.id = id
        self.name = name
        self.gender = gender
        self.age = age
        Student.count += 1
    
    def getAge(self):
        return self.age
    
    def setAge(self, newage):
        self.age = newage

    def showInfo(self):
        return f"id:{self.id}, name:{self.name}, gender:{self.gender}, age:{self.age}"

student1 = Student('202201', '张三', 'male', 18)
print(student1.showInfo())


student2 = Student('202202', '李四', 'female', 20)
print(student2.showInfo())


student1.setAge(19)
print(student1.showInfo())

student3 = Student('202203', '王五', 'female', 17)
print(student3.showInfo())

print("学生人数:", Student.count)