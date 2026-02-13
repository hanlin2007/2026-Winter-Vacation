
# Python 面向对象

## 1️⃣ 类与对象的基本概念

- **类**：像一张设计图纸，描述了一类事物的属性和行为。
    
- **对象**：根据类创建出来的具体实例。

```
class Dog:
    pass

dog1 = Dog()  # dog1 是 Dog 类的一个对象（实例）
```

---

## 2️⃣ 构造方法 `__init__`

### 2.1 作用

`__init__`是类的**构造方法**，在创建对象时自动调用，用来初始化对象的属性。

### 2.2 语法

```
class Dog:
    def __init__(self, name, age):  # self 必须写，其他参数按需添加
        self.name = name   # 成员属性
        self.age = age
```

### 2.3 调用构造方法

```
dog1 = Dog("Buddy", 3)  # 创建对象时传入参数
print(dog1.name)        # 输出 Buddy
```

- 注意：`__init__`的括号里除了 `self`外，可以有任意多个参数，但创建对象时要按顺序传参。

---

## 3️⃣ 成员对象（属性）与成员方法

### 3.1 成员对象（属性）

- 在 `__init__`中用 `self.xxx = ...`定义的变量，是**实例属性**，每个对象独立拥有。
    

```
class Dog:
    def __init__(self, name):
        self.name = name   # 成员属性
```

### 3.2 成员方法

- 定义在类中的函数，第一个参数必须是 `self`，表示当前对象。
    

```
class Dog:
    def bark(self):
        print(f"{self.name} says woof!")
```

### 3.3 方法调用

```
dog1 = Dog("Buddy")
dog1.bark()  # 输出 Buddy says woof!
```

- 调用时不需要传 `self`，Python 会自动把 `dog1`作为 `self`传入。
    

---

## 4️⃣ 继承与 `super()`

### 4.1 继承语法

```
class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):  # Dog 继承 Animal
    pass
```

### 4.2 调用父类构造方法

如果子类有自己的 `__init__`，并且想保留父类的初始化逻辑，需要用 `super()`：

#### 旧式写法（Python 2 风格，Python 3 也兼容）：

```
class Dog(Animal):
    def __init__(self, name, breed):
        super(Dog, self).__init__(name)  # 第一个参数是子类名，第二个是 self
        self.breed = breed
```

#### 新式写法（推荐，Python 3）：

```
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # 不需要传子类名和 self
        self.breed = breed
```

**为什么旧式写法有两个参数？**​

- `super(LeNet, self)`返回一个代理对象，它会按照方法解析顺序（MRO）找到 `LeNet`的下一个类（即 `nn.Module`），然后把 `self`传给它的方法。
    
- 在 Python 3 中，`super()`默认就是 `super(CurrentClass, self)`，所以可以省略参数。
    

---

## 5️⃣ 多态

### 5.1 什么是多态？

多态的意思是：**不同类的对象可以对同一方法做出不同的响应**。

在 Python 中，由于是动态类型，多态是自然支持的，不需要像 C++ 那样声明虚函数。

```
class Cat:
    def speak(self):
        print("Meow")

class Dog:
    def speak(self):
        print("Woof")

def animal_sound(animal):
    animal.speak()  # 不关心具体类型，只要它有 speak 方法

cat = Cat()
dog = Dog()

animal_sound(cat)  # Meow
animal_sound(dog)  # Woof
```

---

## 6️⃣ 鸭子多态（Duck Typing）

### 6.1 概念

> “If it looks like a duck, swims like a duck, and quacks like a duck, then it probably is a duck.”

在 Python 中，**不检查对象的类型，只检查它有没有某个方法或属性**。这就是鸭子多态。

### 6.2 例子

```
class Duck:
    def quack(self):
        print("Quack!")

class Person:
    def quack(self):
        print("I'm quacking like a duck!")

def make_it_quack(obj):
    obj.quack()  # 只要有 quack 方法就能调用

duck = Duck()
person = Person()

make_it_quack(duck)    # Quack!
make_it_quack(person)  # I'm quacking like a duck!
```

- `Person`不是 `Duck`的子类，但因为也有 `quack`方法，所以也能被 `make_it_quack`调用。
    

---

## 7️⃣ 总结

| 概念      | 语法示例                      | 说明               |
| ------- | ------------------------- | ---------------- |
| 类       | `class Dog:`              | 定义类              |
| 对象      | `dog = Dog()`             | 实例化              |
| 构造方法    | `def __init__(self, ...)` | 初始化属性            |
| 成员属性    | `self.name = ...`         | 每个对象独有           |
| 成员方法    | `def bark(self):`         | 第一个参数必须是 self    |
| 方法调用    | `dog.bark()`              | 自动传入 self        |
| 继承      | `class Dog(Animal):`      | 子类继承父类           |
| super() | `super().__init__(...)`   | 调用父类构造方法         |
| 多态      | `animal.speak()`          | 不同类型对象对同一方法有不同实现 |
| 鸭子多态    | `obj.quack()`             | 不关心类型，只关心是否有该方法  |
|         |                           |                  |


---



# PyTorch 框架中的语法解析

PyTorch 中定义神经网络时，通常会继承 `nn.Module`，这就是典型的 **面向对象继承与封装**

```
import torch.nn as nn

class LeNet(nn.Module):           # 1. 继承 nn.Module
    def __init__(self):            # 2. 构造方法
        super(LeNet, self).__init__()  # 调用父类 nn.Module 的构造方法
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):          # 3. 前向传播方法
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

---

逐行解释与 OOP 对应关系

|代码片段|OOP 概念|说明|
|---|---|---|
|`class LeNet(nn.Module):`|**继承**​|`LeNet`继承自 `nn.Module`，获得神经网络模块的功能（如参数管理、GPU迁移等）。|
|`def __init__(self):`|**构造方法**​|初始化网络层，`self.conv1`、`self.conv2`等是实例属性（即网络层对象）。|
|`super(LeNet, self).__init__()`|**调用父类构造方法**​|确保 `nn.Module`正确初始化。|
|`self.conv1 = ...`|**封装**​|将卷积层作为对象的状态（属性）保存。|
|`def forward(self, x):`|**方法重写**​|`nn.Module`要求子类实现 `forward`方法，这是多态的体现——不同网络有不同的前向逻辑。|
|`x = self.conv1(x)`|**方法调用**​|调用 `nn.Conv2d`对象的 `__call__`方法（实际执行 `forward`）。|


```
model = LeNet()          # 创建实例
output = model(input)    # 调用 forward（实际是 model.__call__ 会调用 forward）
```

- `model(input)`能直接运行，是因为 `nn.Module`实现了 `__call__`方法，它内部调用了 `forward`。
    


