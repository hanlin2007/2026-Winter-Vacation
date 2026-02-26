参考了视频[Understanding Dropout (C2W1L07)](https://www.youtube.com/watch?v=ARq74QuavAo)

为何需要Dropout：**过拟合**

训练时，每个神经元以概率p随机丢弃：
$$输出 = \begin{cases} 0, & \text{以概率p} \\ \frac{原值}{1-p}, & \text{以概率1-p} \end{cases}$$

测试时，所有神经元都参与，但输出要乘以(1-p)保持期望一致。


```python
import torch
import torch.nn as nn

class MLPWithDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
        # 🔑 Dropout层
        self.dropout = nn.Dropout(p=dropout_rate)  # p是丢弃概率
        
    def forward(self, x):
        x = self.flatten(x)
        
        # 第一层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # 随机丢弃50%的神经元
        
        # 第二层
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 输出层（通常不加dropout）
        x = self.fc3(x)
        return x

# 训练时的使用
model = MLPWithDropout(dropout_rate=0.5)
model.train()  #训练模式：dropout生效

# 测试时的使用
model.eval()   # 评估模式：dropout自动关闭
with torch.no_grad():  # 不计算梯度
    test_output = model(test_data)

# **重要提示：** 
# PyTorch的Dropout在训练时自动缩放，不需要手动处理！
```

 **实用技巧：不同层用不同dropout率
```python      
# 输入层附近用小dropout，深层用大dropout
self.dropout1 = nn.Dropout(0.2)  # 低层：保留更多信息
self.dropout2 = nn.Dropout(0.3)  
self.dropout3 = nn.Dropout(0.4)  # 高层：更强的正则化
```
