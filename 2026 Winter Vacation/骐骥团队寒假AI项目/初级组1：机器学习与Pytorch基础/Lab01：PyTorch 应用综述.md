
==在深入细节之前，我参考了Pytorch官网的快速入门Youtube系列视频，同时参考了b站的深度学习入门视频，这个笔记中，我对Pytorch从张量到模型训练保存的完整流程有了一个大致了解，同时能够跟着视频从零用Pytorch实现一个卷积神经网络==
## PyTorch 张量  [[Lab02：PyTorch Tensor]]#

视频 [03:50](https://www.youtube.com/watch?v=IC0_FRiX-sw&t=230s) 。

首先我们来看一些基本的张量操作。来看几种创建张量的方法：

```
z = torch.zeros(5, 3)
print(z)
print(z.dtype)
```

```
tensor([[0., 0., 0.],            # tensor类型数据的元组呈现
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
torch.float32                    # data type
```


上面，我们创建了一个 5x3 的矩阵，其中填充了零，并查询其数据类型，发现这些零是**32 位浮点数，这是 PyTorch 的默认数据类型**。

如果你想要的是整数呢？你可以随时覆盖默认值：

```
i = torch.ones((5, 3), dtype=torch.int16)   # 作为tensor可选的参数
print(i)
```

```
tensor([[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]], dtype=torch.int16)
```


你可以看到，当我们更改默认值时，张量在打印时会很**贴心**地报告这一点。

----

通常的做法是**随机初始化学习权重**，为了保证结果的可重复性，通常会为伪随机数生成器（PRNG）设置一个特定的种子：

```
torch.manual_seed(1729)
r1 = torch.rand(2, 2)
print('A random tensor:')
print(r1)

r2 = torch.rand(2, 2)                    # 利用初始化的种子继续第二次生成张量
print('\nA different random tensor:')
print(r2)                                # new values

torch.manual_seed(1729)
r3 = torch.rand(2, 2)
print('\nShould match r1:')
print(r3)                          # repeats values of r1 because of re-seed
```

```
A random tensor:
tensor([[0.3126, 0.3791],
        [0.3087, 0.0736]])

A different random tensor:
tensor([[0.4216, 0.0691],
        [0.2332, 0.4047]])

Should match r1:
tensor([[0.3126, 0.3791],
        [0.3087, 0.0736]])
```

PyTorch 张量能够以直观的方式执行算术运算。形状相似的张量可以进行加法、乘法等运算。标量运算则分布在张量上：

```
ones = torch.ones(2, 3)
print(ones)

twos = torch.ones(2, 3) * 2     # every element is multiplied by 2
print(twos)

threes = ones + twos          # addition allowed because shapes are similar
print(threes)                 # tensors are added element-wise
print(threes.shape)           # this has the same dimensions as input tensors

r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)
# uncomment this line to get a runtime error
# r3 = r1 + r2
```

```
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[2., 2., 2.],
        [2., 2., 2.]])
tensor([[3., 3., 3.],
        [3., 3., 3.]])
torch.Size([2, 3])
```


以下是一些可用的数学运算示例：

```
r = (torch.rand(2, 2) - 0.5) * 2                # 由随机数导出-1~1范围的张量元素
print('A random matrix, r:')
print(r)

# Common mathematical operations are supported:       # 绝对值
print('\nAbsolute value of r:')
print(torch.abs(r))

# trigonometric functions:                  # 反三角函数运算
print('\nInverse sine of r:')
print(torch.asin(r))

# linear algebra operations like determinant and singular value decomposition
print('\nDeterminant of r:')
print(torch.det(r))
print('\nSingular value decomposition of r:')
print(torch.svd(r))

# statistical and aggregate operations:
print('\nAverage and standard deviation of r:')
print(torch.std_mean(r))
print('\nMaximum value of r:')
print(torch.max(r))
```

```
A random matrix, r:
tensor([[ 0.9956, -0.2232],
        [ 0.3858, -0.6593]])

Absolute value of r:
tensor([[0.9956, 0.2232],
        [0.3858, 0.6593]])

Inverse sine of r:
tensor([[ 1.4775, -0.2251],
        [ 0.3961, -0.7199]])

Determinant of r:
tensor(-0.5703)

Singular value decomposition of r:
torch.return_types.svd(
U=tensor([[-0.8353, -0.5497],
        [-0.5497,  0.8353]]),
S=tensor([1.1793, 0.4836]),
V=tensor([[-0.8851, -0.4654],
        [ 0.4654, -0.8851]]))

Average and standard deviation of r:
(tensor(0.7217), tensor(0.1247))

Maximum value of r:
tensor(0.9956)
```

关于 PyTorch 张量的强大功能还有很多值得了解的地方，包括如何设置它们以在 GPU 上进行并行计算。




----



## PyTorch 模型#

视频 [10:00](https://www.youtube.com/watch?v=IC0_FRiX-sw&t=600s)。

如何在 PyTorch 中表达模型：

```
import torch                     # for all things PyTorch
import torch.nn as nn            # neural network
import torch.nn.functional as F  # for the activation function 激活层函数
```


![le-net-5 diagram](https://docs.pytorch.org/tutorials/_images/mnist.png)


上图是 LeNet-5 的示意图，它是最早的卷积神经网络之一，也是深度学习爆发式增长的驱动力之一。它被设计用于读取手写数字的小图像（MNIST 数据集），并正确分类图像中表示的是哪个数字。

Lenet5的结构示意图详解
1.图中包括两个卷积层，两个池化层，三个全连接层
2.灰色方块表示张量经过层后的输出特征图，卷积核大小、输入输出通道数和尺寸大小

以下是其工作原理的简要说明：
- C1 层是一个卷积层，它会扫描输入图像，寻找训练过程中学习到的特征。它输出一个激活图，显示每个学习到的特征在图像中出现的位置。这个“激活图”在 S2 层进行下采样。
- C3 层是另一个卷积层，这次它扫描 C1 层的激活图，寻找特征 *组合* 。它还会输出一个描述这些特征组合空间位置的激活图，该激活图在 S4 层进行下采样。
- 最后，末端的全连接层 F5、F6 和 OUTPUT 是一个 *分类器* ，它接收最终的激活图，并将其分类到代表 10 个数字的 10 个箱之一中。

我们如何用代码表达这个简单的神经网络？

```
class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # an affine operation: y = Wx + b  仿射算子
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
    
        # Max pooling over a (2, 2) window
        # import torch.nn.functional as F (for the activation function)
        
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        x = x.view(-1, self.num_flat_features(x))
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]      # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

仔细查看这段代码，你应该能够发现它与上面的图表有一些结构上的相似之处。

这展示了一个典型的 PyTorch 模型的结构：

- 继承自 `torch.nn.Module` 模块可以嵌套——事实上，甚至 `Conv2d` 和 `Linear` 层类也继承自 `torch.nn.Module` 。
- 模型将有一个 `__init__()` 函数，它会在其中实例化其层，并加载它可能需要的任何数据工件（例如，NLP 模型可能会加载词汇表）。
- 模型会包含一个 `forward()` 函数。实际的计算过程就在这里进行：输入会依次经过网络层和各种函数，最终生成输出。

让我们实例化这个对象，并向它输入一个示例数据。

```
net = LeNet()
print(net)                         # what does the object tell us about itself?

input = torch.rand(1, 1, 32, 32)   # stand-in for a 32x32 black & white image
print('\nImage batch shape:')
print(input.shape)

output = net(input)                # we don't call forward() directly
print('\nRaw output:')
print(output)
print(output.shape)
```

```
LeNet(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)

Image batch shape:
torch.Size([1, 1, 32, 32])

Raw output:
tensor([[ 0.0898,  0.0318,  0.1485,  0.0301, -0.0085, -0.1135, -0.0296,  0.0164,
          0.0039,  0.0616]], grad_fn=<AddmmBackward0>)
torch.Size([1, 10])
```

以上内容包含几个重要信息：

首先，我们实例化 `LeNet` 类，并打印 `net`，`torch.nn.Module` 的子类对象会报告它创建的层及其形状和参数。如果您想了解模型的处理过程，这可以提供一个方便的概览。

下面，我们创建一个虚拟输入，代表一个 32x32 像素、单通道的图像。通常情况下，你可以加载一个图像图块并将其转换为这种形状的张量。

您可能已经注意到我们的张量多了一个维度—— *批次维度。PyTorch* 模型假定它们处理的是*批次*数据。 例如，一批 16 个图像图块的形状为 `(16, 1, 32, 32)` 由于我们只使用一张图像，因此我们创建一个形状为 `(1, 1, 32, 32)` 批次，数量为 1。

我们像调用函数一样调用模型，请求它进行推理： `net(input)` 的输出表示模型对输入代表特定数字的置信度。（由于该模型实例尚未学习任何内容，我们不应期望在输出中看到任何信号。）观察 `output` 的形状，我们可以看到它也具有批次维度，其大小应始终与输入批次维度匹配。如果我们传入的输入批次包含 16 个实例， `output` 形状将为 `(16, 10)` 。



----


## 从零设计和实现一个卷积神经网络



![le-net-5 diagram](https://docs.pytorch.org/tutorials/_images/mnist.png)

```
import torch
from torch import nn

#定义张量x，它的尺寸是5*1*28*28
#表示了5个，单通道，28*28的数据
x = torch.zeros([5,1,28,28])

#定义一个输入通道是1，输出通道是6，卷积核大小是5*5的卷积层
conv = nn.Conv2d(in channels=1,out channels=6,kernel size=5)

#将x，输入至conv，计算出结果c
c = conv(x)

#打印结果尺寸                  #程序输出结果
print(c.shape)               torch.Size([5,6,24,24])

#定义最大池化层
pool = nn.MaxPool2d(2)

#将卷积层计算得到的特征图c，输入至pool
s = pool(c)

#输出s的尺寸                  #程序输出结果
print(s.shape)              torch.Size([5,6,12,12])
```


**使用Pytorch框架完整实现Lenet5网络**

```
import torch
from torch.nn import Module
from torch import nn

# 使用Pytorch，完整实现Lenet5网络
class Lenet(Module):

    # Lenet5网络结构的定义
    def__init__(self):
    
    super(Lenet5,self).__init__()
    
    # 定义Lenet5模型中的结构
    # 第一个卷积块
    self.conv1 = nn.Sequential(
        nn.Conv2d(1,6,5),  # 卷积层
        nn.ReLU(),         # relu激活函数
        nn.MaxPool2d(2)    # 最大池化层
    )
    # 第二个卷积块
    self.conv2 = nn.Sequential(
        nn.Conv2d(6,16,5), # 卷积层
        nn.ReLU(),         # relu激活函数
        nn.MaxPool2d(2)    # 最大池化层
    )
    
    # 全连接层1
    self.fc1 = nn.Sequential(
        nn.Linear(256,120) # 线性层
        nn.ReLU            # relu激活函数
    )
    # 全连接层2
    self.fc1 = nn.Sequential(
        nn.Linear(120,84)  # 线性层
        nn.ReLU            # relu激活函数
    )
    # 全连接层3
    self.fc1 = nn.Sequential(
        nn.Linear(84,10)   # 线性层
    )
    
    # 前向传播计算函数，输入张量x
    def forward(self,x):
    # 在注释中，标记了张量x经过每一层计算之后变化的尺寸
    # 最初输入：n*1*28*28    
    x = self.conv1(x)      # [n,6,12,12]
    x = self.conv2(x)      # [n,16,4,4]
    
    # 在输入至全连接层前
    # 需要使用view函数，将张量的维度从n*16*4*4展平为n*256
    x = x.view(-1,256)     # [n,256]
    x = self.fc1(x)        # [n,120]
    x = self.fc2(x)        # [n,84]
    x = self.fc3(x)        # [n,10]
    return x

```

**使用Lenet5，Pytorch框架，训练“图像分类”模型
```
import torch
from torch import nn
from torch import optim

from lenet5 import LeNet5

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # 图像预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图
        transforms.ToTensor()     # 转换为张量
    ])
    
    # 读入并构造数据集
    train dataset = datasets.ImageFolder(root='./mnist_images/train',
    transform = transform)
    print("train_dataset length:",len(train_dataset))
    
    # 小批量的数据读入
    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    print("train_loader length:",len(train_loader))
    
    # 创建三个关键对象
    model = LeNet5()        # 前面定义的Lenet5卷积神经网路结构
    optimizer = optim.Adam(model.parameters())   # 优化器
    criterion = nn.CrossEntropyLoss()            # 交叉熵损失误差
    
        
    # 进入模型的迭代循环
    for epoch in range(10):     # 外层循环，表示整个训练数据集的遍历次数
        # 内层循环使用train_loader，进行小批量的数据读取
        for batch_idx,(data,label) in enumerate (train_loader):
        # 每次内层循环，都会进行一次梯度下降算法
        # 梯度下降算法，包括5个步骤：
        output = model(data)    # 1.计算神经网路的前向传播结果
        loss = criterion(output,label)  # 2.计算output和label之间的loss
        loss.backward()         # 3.使用backward计算梯度
        optimizer.step()        # 4.使用optimizer.step更新参数
        optimizer.zero_grad()   # 5.将梯度清零
        
        # 每迭代100个小批量，就打印一次模型的损失，观察训练的过程
        if batch_idx % 100 == 0;
            print(f"Epoch {epoch+1}/10"
                  f"| Batch {batch_idx} / {len(train_loader)}"
                  f"| Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(),'mnist_lenet5.pth')   # 保存模型
```

完成模型训练后，再进行模型的测试




----



## 数据集和数据加载器#

视频 [14:00](https://www.youtube.com/watch?v=IC0_FRiX-sw&t=840s) 

下面演示如何使用 TorchVision 提供的可直接下载的开放获取数据集之一，如何转换图像以供模型使用，以及如何使用 DataLoader 将批量数据提供给模型。

我们首先需要做的是将输入的图像转换为 PyTorch 张量。

```
#%matplotlib inline

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
```

这里，我们为输入数据指定了两种转换：

- `transforms.ToTensor()` 将 Pillow 加载的图像转换为 PyTorch 张量。

- `transforms.Normalize()` 函数会调整张量的值，使其均值为零，标准差为 1.0。大多数激活函数在 x = 0 附近梯度最强，因此将数据中心化到该位置可以加快学习速度。传递给 `transform` 函数的值是数据集中图像 RGB 值的均值（第一个元组）和标准差（第二个元组）。可以通过运行以下几行代码自行计算这些值：

  ```
  from torch.utils.data import ConcatDataset
  transform = transforms.Compose([transforms.ToTensor()])
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                  download=True, transform=transform)
  
  # stack all train images together into a tensor of shape
  # (50000, 3, 32, 32)
  x = torch.stack([sample[0] for sample in ConcatDataset([trainset])])
  
  # get the mean of each channel
  mean = torch.mean(x, dim=(0,2,3))     # tensor([0.4914, 0.4822, 0.4465])
  std = torch.std(x, dim=(0,2,3))       # tensor([0.2470, 0.2435, 0.2616])
  ```

还有许多其他变换方式可供选择，包括裁剪、居中、旋转和反射。

接下来，我们将创建一个 CIFAR10 数据集实例。这是一个包含 32x32 像素彩色图像块的数据集，代表 10 类对象：6 类动物（鸟、猫、鹿、狗、青蛙、马）和 4 类车辆（飞机、汽车、轮船、卡车）。
```
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
```

```
  0%|          | 0.00/170M [00:00<?, ?B/s]
  0%|          | 459k/170M [00:00<00:37, 4.55MB/s]
  4%|▍         | 7.63M/170M [00:00<00:03, 43.9MB/s]
 11%|█         | 19.0M/170M [00:00<00:02, 75.5MB/s]
 18%|█▊        | 30.3M/170M [00:00<00:01, 90.5MB/s]
 24%|██▍       | 41.4M/170M [00:00<00:01, 97.6MB/s]
 31%|███       | 52.7M/170M [00:00<00:01, 103MB/s]
 37%|███▋      | 63.8M/170M [00:00<00:01, 105MB/s]
 44%|████▍     | 75.1M/170M [00:00<00:00, 108MB/s]
 51%|█████     | 86.4M/170M [00:00<00:00, 110MB/s]
 57%|█████▋    | 97.8M/170M [00:01<00:00, 111MB/s]
 64%|██████▍   | 109M/170M [00:01<00:00, 112MB/s]
 71%|███████   | 121M/170M [00:01<00:00, 112MB/s]
 77%|███████▋  | 132M/170M [00:01<00:00, 113MB/s]
 84%|████████▍ | 143M/170M [00:01<00:00, 113MB/s]
 91%|█████████ | 155M/170M [00:01<00:00, 113MB/s]
 97%|█████████▋| 166M/170M [00:01<00:00, 112MB/s]
100%|██████████| 170M/170M [00:01<00:00, 104MB/s]
```

这是一个在 PyTorch 中创建数据集对象的示例。可下载的数据集（例如上文提到的 CIFAR-10）是以下类别的子类： `torch.utils.data.Dataset` 中的 `Dataset` 类包括 TorchVision、Torchtext 和 TorchAudio 中的可下载数据集，以及诸如 `torchvision.datasets.ImageFolder` 之类的实用数据集类，它可以读取一个包含已标注图像的文件夹。您还可以创建自己的 `Dataset` 子类。


当我们实例化数据集时，需要告诉它一些信息：

- 数据存放的文件系统路径。
- 无论我们是否将此数据集用于训练；大多数数据集都会被分成训练子集和测试子集。
- 是否要下载数据集（如果尚未下载）。
- 我们希望对数据进行以下转换。

数据集准备就绪后，即可将其交给 `DataLoader` 器：

```
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
```

`Dataset` 子类封装了对数据的访问，并针对其提供的数据类型进行了专门化。DataLoader 对数据本身*一无所知* ，它只是根据指定的参数 `DataLoader` 将 Dataset 提供的输入张量组织成批次。

在上面的示例中，我们要求 `DataLoader` 从 `trainset` 提供 4 张图像的批次，并随机化它们的顺序（ `shuffle=True` ），我们还告诉它启动两个工作进程从磁盘加载数据。

将 `DataLoader` 处理的批次可视化是一种很好的做法：

```
import matplotlib.pyplot as plt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```
![[Pasted image 20260211200151.png]]
```
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-0.49473685..1.5632443].

 ship   car   horse    ship
```

运行上述单元格应该会显示四张图片，以及每张图片的正确标签。




----




## 训练你的 PyTorch 模型#

视频 [17:10](https://www.youtube.com/watch?v=IC0_FRiX-sw&t=1030s)

让我们把所有要素整合起来，训练一个模型：

```
#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

首先，我们需要训练集和测试集。如果您尚未下载数据集，请运行下面的单元格以确保数据集已下载。（可能需要一分钟。）

```
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

我们将对 `DataLoader` 的输出进行检查：

```
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```
![[Pasted image 20260211200403.png]]
`cat   cat  deer  frog`




下面就是我们**将要训练的模型**。如果它看起来很眼熟，那是因为它是 LeNet 的一个变体——LeNet前面已经讨论过——专门针对三色图像进行了调整。

```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
```


最后我们需要的是损失函数和优化器：

```
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

损失函数衡量的是模型预测结果与理想输出之间的差距。交叉熵损失是像我们这样的分类模型常用的损失函数。

**优化器**是驱动学习的关键。这里我们创建了一个实现*随机梯度下降的优化器，随机梯度下降*是一种比较直接的优化算法。除了算法的参数，例如学习率（ `lr` ）和动量之外，我们还传入了`net.parameters()` ，它是模型中所有学习权重的集合——优化器会调整这些权重。

最后，所有这些步骤都整合到训练循环中。运行此单元格，执行可能需要几分钟时间：

```
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

这里，我们只进行了 **2 个训练周期** （第 1 行）——两次循环 遍历训练数据集。每次遍历都有一个内部循环，该内部循环 **遍历训练数据** （第 4 行），提供一批批转换后的输入图像及其正确的标签。

**将梯度归零** （第 9 行）是一个重要的步骤。梯度是在一个批次内累积的；如果我们不为每个批次重置梯度，梯度就会持续累积，从而导致梯度值不正确，使学习无法进行。

在第 12 行，我们**请求模型对该批次数据进行预测** 。 在下一行（13）中，我们计算损失——即两者之间的差值 `outputs` （模型预测）和 `labels` （正确输出）。

在第 14 行，我们执行 `backward()` 传递，并计算将指导学习的梯度。

在第 15 行，优化器执行一个学习步骤——它使用 `backward()` 调用中的梯度来调整学习权重，使其朝着它认为可以减少损失的方向移动。

**运行上面的单元格后，** 应该会看到类似这样的内容：

```
[1,  2000] loss: 2.235
[1,  4000] loss: 1.940
[1,  6000] loss: 1.713
[1,  8000] loss: 1.573
[1, 10000] loss: 1.507
[1, 12000] loss: 1.442
[2,  2000] loss: 1.378
[2,  4000] loss: 1.364
[2,  6000] loss: 1.349
[2,  8000] loss: 1.319
[2, 10000] loss: 1.284
[2, 12000] loss: 1.267
Finished Training
```

请注意，损失是单调递减的，这表明我们的模型在训练数据集上的性能正在不断提高。

最后一步，我们应该检查模型是否真的在运行。 这指的是模型进行**通用学习**，而不仅仅是“记忆”数据集。这种情况称为 **过拟合，** 通常表明数据集太小（样本不足以进行通用学习），或者模型的学习参数过多，无法正确建模数据集。

这就是数据集被分成训练集和测试集的原因——为了测试模型的通用性，我们要求它对未训练过的数据进行预测：

```
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```

```
Accuracy of the network on the 10000 test images: 54 %
```

模型目前的准确率大约为 50%。这虽然算不上最先进的模型，但远高于随机输出预期的 10% 准确率。这表明模型确实进行了一些通用学习。