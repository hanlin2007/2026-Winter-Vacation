这段代码也是完完全全Claude写的，我只是学了学AI的代码，然后在本地试着跑了一跑。其实也有挺多思考和感触的。

第一是，我的电脑运行这个程序用了差不多30多分钟吧，而且还使用的是CPU双进程加载、GPU训练模型，就这样也能感受到真的挺需要算力和资源的，这是第一次感受这么真切。（而且越训练效果越差越波动是真的很恼火哈哈哈，训练论述、学习率这种参数真的很棘手呀）

另外呢，我观察进度条能明显感觉到，在每一轮Epoch开始和结束之后，风扇猛转，耗时特别长，我想起了之前做过的一个存储项目，或许是加载数据集的时候磁盘IO时间限制了整个模型训练和测试的性能！！这或许又与我之前“分层存储+深度学习”的项目有微妙的联系，不过这是存储方向上的启发。

最后，不知道还能不能看到这里。随着最后一个项目的运行完成，一整个寒假的pytorch学习和付出也算是终于结束，这个过程中真的有很多感触，第一次实打实走进深度学习，第一次走到人工智能的背后，或许ai还会越来越强，但我想，比ai取代程序员更可怕的永远是程序员失去思考。感谢骐骥团队在这个寒假给我的这次学习机会，期待骐骥团队见！！

```python
"""
CIFAR-10 物体识别
类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 数据增强（CIFAR-10需要更强的数据增强）
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.RandomHorizontalFlip(),      # 随机翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载CIFAR-10
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)

# 划分验证集
train_size = 45000
val_size = 5000
trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=2)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# 构建一个更强的CNN模型（类似VGG但适合小图片）
class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 32x32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            
            # Block 2: 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            
            # Block 3: 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 训练CIFAR-10模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CIFAR10CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 训练循环
def train_cifar10(model, trainloader, valloader, epochs=50):
    best_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # 记录
        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss/len(trainloader))
        val_losses.append(val_loss/len(valloader))
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 学习率调整
        scheduler.step()
        
        print(f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'cifar10_best.pth')
            print(f'✓ 保存最佳模型，准确率: {val_acc:.2f}%')
    
    return train_losses, val_losses, train_accs, val_accs

# 开始训练
history = train_cifar10(model, trainloader, valloader, epochs=50)

# 测试
model.load_state_dict(torch.load('cifar10_best.pth'))
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        test_total += targets.size(0)
        test_correct += predicted.eq(targets).sum().item()

print(f'\n测试集准确率: {100. * test_correct / test_total:.2f}%')
```

```bash
PS C:\Users\lihanlin2007\Desktop\pytorch_learning> & C:\Users\lihanlin2007\anaconda3\envs\pytorch\python.exe c:/Users/lihanlin2007/Desktop/pytorch_learning/cifar-10.py
使用设备: cuda
Epoch 1/50: 100%|████████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.21it/s, Loss=1.315, Acc=29.43%]
Train Acc: 29.43%, Val Acc: 35.96%, LR: 0.099994
✓ 保存最佳模型，准确率: 35.96%
Epoch 2/50: 100%|████████████████████████████████████████████████████████████████████████████| 352/352 [00:21<00:00, 16.29it/s, Loss=0.968, Acc=51.83%]
Train Acc: 51.83%, Val Acc: 58.52%, LR: 0.099975
✓ 保存最佳模型，准确率: 58.52%
Epoch 3/50: 100%|████████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 16.82it/s, Loss=1.083, Acc=62.52%]
Train Acc: 62.52%, Val Acc: 57.00%, LR: 0.099944
Epoch 4/50: 100%|████████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.10it/s, Loss=1.056, Acc=68.28%]
Train Acc: 68.28%, Val Acc: 58.42%, LR: 0.099901
Epoch 5/50: 100%|████████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.16it/s, Loss=0.907, Acc=72.69%]
Train Acc: 72.69%, Val Acc: 65.28%, LR: 0.099846
✓ 保存最佳模型，准确率: 65.28%
Epoch 6/50: 100%|████████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 16.92it/s, Loss=0.584, Acc=75.60%] 
Train Acc: 75.60%, Val Acc: 73.66%, LR: 0.099778
✓ 保存最佳模型，准确率: 73.66%
Epoch 7/50: 100%|████████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.01it/s, Loss=0.474, Acc=77.44%] 
Train Acc: 77.44%, Val Acc: 73.40%, LR: 0.099698
Epoch 8/50: 100%|████████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 16.95it/s, Loss=0.558, Acc=78.52%]
Train Acc: 78.52%, Val Acc: 72.72%, LR: 0.099606
Epoch 9/50: 100%|████████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.09it/s, Loss=0.653, Acc=79.78%]
Train Acc: 79.78%, Val Acc: 77.94%, LR: 0.099501
✓ 保存最佳模型，准确率: 77.94%
Epoch 10/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.18it/s, Loss=0.550, Acc=80.63%]
Train Acc: 80.63%, Val Acc: 77.16%, LR: 0.099384
Epoch 11/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.14it/s, Loss=0.540, Acc=80.62%]
Train Acc: 80.62%, Val Acc: 76.28%, LR: 0.099255
Epoch 12/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.07it/s, Loss=0.511, Acc=81.84%]
Train Acc: 81.84%, Val Acc: 73.62%, LR: 0.099114
Epoch 13/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.21it/s, Loss=0.427, Acc=82.27%]
Train Acc: 82.27%, Val Acc: 70.00%, LR: 0.098961
Epoch 14/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.15it/s, Loss=0.402, Acc=82.68%]
Train Acc: 82.68%, Val Acc: 80.62%, LR: 0.098796
✓ 保存最佳模型，准确率: 80.62%
Epoch 15/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.06it/s, Loss=0.530, Acc=82.86%] 
Train Acc: 82.86%, Val Acc: 82.82%, LR: 0.098618
✓ 保存最佳模型，准确率: 82.82%
Epoch 16/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.04it/s, Loss=0.530, Acc=83.06%] 
Train Acc: 83.06%, Val Acc: 69.52%, LR: 0.098429
Epoch 17/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.12it/s, Loss=0.408, Acc=83.65%]
Train Acc: 83.65%, Val Acc: 79.48%, LR: 0.098228
Epoch 18/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.02it/s, Loss=0.402, Acc=83.71%]
Train Acc: 83.71%, Val Acc: 80.62%, LR: 0.098015
Epoch 19/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 16.99it/s, Loss=0.485, Acc=84.04%]
Train Acc: 84.04%, Val Acc: 80.12%, LR: 0.097790
Epoch 20/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 16.98it/s, Loss=0.469, Acc=84.23%]
Train Acc: 84.23%, Val Acc: 83.54%, LR: 0.097553
✓ 保存最佳模型，准确率: 83.54%
Epoch 21/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.04it/s, Loss=0.616, Acc=84.48%] 
Train Acc: 84.48%, Val Acc: 78.40%, LR: 0.097304
Epoch 22/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.03it/s, Loss=0.571, Acc=84.63%]
Train Acc: 84.63%, Val Acc: 82.82%, LR: 0.097044
Epoch 23/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 16.80it/s, Loss=0.520, Acc=84.57%]
Train Acc: 84.57%, Val Acc: 77.18%, LR: 0.096772
Epoch 24/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.14it/s, Loss=0.532, Acc=84.68%]
Train Acc: 84.68%, Val Acc: 79.38%, LR: 0.096489
Epoch 25/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.23it/s, Loss=0.523, Acc=85.16%]
Train Acc: 85.16%, Val Acc: 74.84%, LR: 0.096194
Epoch 26/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.13it/s, Loss=0.373, Acc=85.13%]
Train Acc: 85.13%, Val Acc: 80.88%, LR: 0.095888
Epoch 27/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.15it/s, Loss=0.418, Acc=85.20%]
Train Acc: 85.20%, Val Acc: 79.78%, LR: 0.095570
Epoch 28/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 16.96it/s, Loss=0.255, Acc=85.63%]
Train Acc: 85.63%, Val Acc: 82.78%, LR: 0.095241
Epoch 29/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.19it/s, Loss=0.330, Acc=85.57%]
Train Acc: 85.57%, Val Acc: 77.74%, LR: 0.094901
Epoch 30/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.02it/s, Loss=0.533, Acc=85.50%]
Train Acc: 85.50%, Val Acc: 83.60%, LR: 0.094550
✓ 保存最佳模型，准确率: 83.60%
Epoch 31/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.15it/s, Loss=0.355, Acc=85.91%] 
Train Acc: 85.91%, Val Acc: 83.30%, LR: 0.094188
Epoch 32/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.22it/s, Loss=0.265, Acc=85.91%]
Train Acc: 85.91%, Val Acc: 79.56%, LR: 0.093815
Epoch 33/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.18it/s, Loss=0.642, Acc=85.64%]
Train Acc: 85.64%, Val Acc: 74.88%, LR: 0.093432
Epoch 34/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.14it/s, Loss=0.492, Acc=85.81%]
Train Acc: 85.81%, Val Acc: 83.22%, LR: 0.093037
Epoch 35/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.06it/s, Loss=0.512, Acc=86.11%]
Train Acc: 86.11%, Val Acc: 79.40%, LR: 0.092632
Epoch 36/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.02it/s, Loss=0.552, Acc=86.00%]
Train Acc: 86.00%, Val Acc: 83.50%, LR: 0.092216
Epoch 37/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.16it/s, Loss=0.601, Acc=86.25%]
Train Acc: 86.25%, Val Acc: 80.14%, LR: 0.091790
Epoch 38/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.18it/s, Loss=0.516, Acc=86.01%]
Train Acc: 86.01%, Val Acc: 80.60%, LR: 0.091354
Epoch 39/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 16.99it/s, Loss=0.726, Acc=86.68%]
Train Acc: 86.68%, Val Acc: 81.62%, LR: 0.090907
Epoch 40/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 16.97it/s, Loss=0.373, Acc=86.24%]
Train Acc: 86.24%, Val Acc: 78.92%, LR: 0.090451
Epoch 41/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.23it/s, Loss=0.444, Acc=86.81%]
Train Acc: 86.81%, Val Acc: 83.56%, LR: 0.089984
Epoch 42/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.14it/s, Loss=0.370, Acc=86.82%]
Train Acc: 86.82%, Val Acc: 80.12%, LR: 0.089508
Epoch 43/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.21it/s, Loss=0.318, Acc=86.28%]
Train Acc: 86.28%, Val Acc: 82.10%, LR: 0.089022
Epoch 44/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.02it/s, Loss=0.373, Acc=86.82%]
Train Acc: 86.82%, Val Acc: 85.46%, LR: 0.088526
✓ 保存最佳模型，准确率: 85.46%
Epoch 45/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 16.95it/s, Loss=0.398, Acc=86.90%]
Train Acc: 86.90%, Val Acc: 84.28%, LR: 0.088020
Epoch 46/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.17it/s, Loss=0.469, Acc=86.90%]
Train Acc: 86.90%, Val Acc: 79.06%, LR: 0.087506
Epoch 47/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.04it/s, Loss=0.386, Acc=86.76%]
Train Acc: 86.76%, Val Acc: 77.40%, LR: 0.086982
Epoch 48/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.22it/s, Loss=0.308, Acc=86.80%]
Train Acc: 86.80%, Val Acc: 84.94%, LR: 0.086448
Epoch 49/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.24it/s, Loss=0.257, Acc=87.36%]
Train Acc: 87.36%, Val Acc: 82.70%, LR: 0.085906
Epoch 50/50: 100%|███████████████████████████████████████████████████████████████████████████| 352/352 [00:20<00:00, 17.00it/s, Loss=0.446, Acc=86.86%]
Train Acc: 86.86%, Val Acc: 80.90%, LR: 0.085355

测试集准确率: 86.08%
```