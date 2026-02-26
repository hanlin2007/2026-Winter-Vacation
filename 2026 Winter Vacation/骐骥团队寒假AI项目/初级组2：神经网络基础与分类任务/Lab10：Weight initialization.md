
以下记录的一些笔记基于Youtube视频[Weight Initialization for Deep Feedforward Neural Networks](https://www.youtube.com/watch?v=tYFO434Lpm0)

**Glorot（Xavier）初始化**
```python
# 适用于linear、tanh、softmax、logistic激活函数
import torch.nn.init as init

linear = nn.Linear(784, 256)
init.xavier_uniform_(linear.weight)
```
**原理：** 让每一层的输入和输出的方差保持一致。

**He初始化**
```python
# 适用于ReLU及其变体
linear = nn.Linear(784, 256)
init.kaiming_uniform_(linear.weight, mode='fan_in', nonlinearity='relu')
```
**原理：** 针对ReLU的特性（一半神经元死亡），将方差扩大一倍。

**Lecun初始化** ：SELU激活函数

