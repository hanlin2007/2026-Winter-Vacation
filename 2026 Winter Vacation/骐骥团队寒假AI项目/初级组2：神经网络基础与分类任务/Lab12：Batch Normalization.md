以下笔记基于Youtube视频[Batch normalization | What it is and how to implement it](https://www.youtube.com/watch?v=yXOMHOpbon8)和AI生成的细节补充

#### 原理：内部协变量偏移

**通俗解释：**
你训练一个模型，每层的数据分布都在变化，导致后一层要不断适应新的分布。就像传送带上物品大小忽大忽小，后面的工人很难操作。BatchNorm就是把数据"标准化"到统一尺度。

**数学原理：**
对每个mini-batch：
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

- $\mu_B$: batch均值
- $\sigma_B^2$: batch方差
- $\gamma, \beta$: 可学习参数（缩放和平移）


```python
import torch
import torch.nn as nn

class MLPWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
        # BatchNorm层
        self.bn1 = nn.BatchNorm1d(256)  # 参数是特征维度
        self.bn2 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        # 顺序：线性层 → BN → 激活函数（经验证，有时激活在前在后效果不同）
        x = self.fc1(x)
        x = self.bn1(x)      # BN在激活前
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.fc3(x)      # 输出层通常不加BN
        return x

# **另一种常见顺序（激活后BN）**
class MLPWithBN_Alt(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.bn1(x)      # BN在激活后
        
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        
        return self.fc3(x)
```

