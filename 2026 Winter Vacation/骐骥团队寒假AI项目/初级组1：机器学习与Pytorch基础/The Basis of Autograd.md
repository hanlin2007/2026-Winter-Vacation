### 数学和机器学习术语：

differentiation：微分
partial derivatives：偏导数
**autograd**：自动求微分
gradient-based optimization：梯度优化

[The Fundamentation Of Autograd](https://www.youtube.com/watch?v=M0fX15_-xrY)
[What is Automatic Differentiation?](https://www.youtube.com/watch?v=wG_nF1awSSY)

铺垫：什么是自动求微分？

微积分中的手动求导数 / 符号微分
数值微分方法，有限差分法来用导数定义求近似
问题：表达式增长，计算参数数量大

自动求微分的两张方式：正向求和反向求



----



PyTorch 的自动微分功能是 PyTorch 灵活性的来源之一， 快速构建机器学习项目。它能够快速且 易于计算多个偏导数（也称为 *梯度）* 经过复杂的计算。此操作是基于反向传播的神经网络学习的核心。

自动微分 (autograd) 的强大之处在于它能 *在运行时动态追踪计算过程。* 这意味着，即使模型包含决策分支或循环（其长度在运行时才能确定），计算过程依然能够被正确追踪，从而获得正确的梯度来驱动学习。此外，由于模型是用 Python 构建的，因此相比那些依赖于对结构更为固定的模型进行静态分析来计算梯度的框架，自动微分提供了更大的灵活性。

## 我们需要自动微分做什么？

机器学习模型是一个*函数* ，它有输入和输出。为了便于讨论，我们将输入视为一个 *i* 维向量。 x⃗*x* ，元素为𝑥 𝑖 x i然后，我们可以将模型 *M* 表示为输入的向量值函数： y⃗=M⃗(x⃗)*y*=*M*(*x*) 。（我们将 M 的输出值视为向量，因为一般来说，一个模型可以有任意数量的输出。）

由于我们主要会在训练的背景下讨论自动微分， 我们感兴趣的输出是模型的损失。 *损失函数* L(𝑦 ⃗ y) = L( M⃗*M* ( x⃗*x* )) 是模型输出的单值标量函数。该函数表示模型预测值与特定输入值的*理想*输出值之间的偏差程度。 *注意：在此之后，我们通常会在上下文明确的情况下省略向量符号，例如，* y*y* 而不是 𝑦 ⃗ y.

在训练模型时，我们希望最小化损失。理想情况下，对于完美模型，这意味着调整其学习权重（即函数的可调参数），使所有输入对应的损失均为零。但在实际应用中，这意味着需要迭代地微调学习权重，直到我们能够针对各种输入获得可接受的损失值。

我们如何决定调整配重的幅度以及方向？ 希望将损失*最小化* ，这意味着要求其一阶导数。 当输入等于 0 时： ∂L∂x=0∂*x*∂*L*=0 。

不过，需要注意的是，损失函数并非*直接*来源于输入，而是模型输出的函数（而模型输出又是输入的直接函数），即 ∂L∂x∂*x*∂*L* = ∂L(y⃗)∂x∂*x*∂*L*(*y*) 。根据微分方程的链式法则，我们有 ∂L(y⃗)∂x∂*x*∂*L*(*y*) = ∂L∂y∂y∂x∂*y*∂*L*∂*x*∂*y* = ∂ 𝐿 ∂ 𝑦 ∂ 𝑀 ( 𝑥 ) ∂ 𝑥 ∂y ∂L∂x ∂M(x).

∂ 𝑀 ( 𝑥 ) ∂ 𝑥 ∂x ∂M(x)事情变得复杂起来。如果我们再次使用链式法则展开表达式，模型输出对其输入的偏导数将涉及对每个乘积学习权重、每个激活函数以及模型中所有其他数学变换的大量局部偏导数。每个此类偏导数的完整表达式是计算图中*所有可能路径*的局部梯度乘积之和，这些路径最终都指向我们试图测量其梯度的变量。

特别是，我们感兴趣的是学习权重上的梯度——它们告诉我们 *应该朝哪个方向改变每个权重* ，才能使损失函数更接近于零。

由于局部导数（每个局部导数对应于模型计算图中的一条独立路径）的数量会随着神经网络深度的增加呈指数级增长，因此计算它们的复杂度也会随之增加。这时，自动微分（autograd）就派上了用场：它跟踪每次计算的历史记录。PyTorch 模型中每个计算出的张量都包含其输入张量以及用于创建该张量的函数的历史记录。此外，PyTorch 中用于作用于张量的函数都内置了计算自身导数的实现，这极大地加快了学习所需的局部导数的计算速度。

## 一个简单的例子#

理论讲了很多——但是自动微分在实践中是什么样的呢？

让我们从一个简单的例子开始。首先，我们需要导入一些模块，以便绘制结果图表：

```
# %matplotlib inline

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
```



接下来，我们将创建一个输入张量，其中包含均匀分布的值。 设定区间为 [0,2π][0,2*π*] ，并指定 `requires_grad=True` 。（与大多数创建张量的函数一样， `torch.linspace()` 接受一个可选的 `requires_grad` 选项。）设置此标志意味着在后续的每次计算中，自动微分都会将计算历史累积到该次计算的输出张量中。

```
a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
print(a)
```



```
tensor([0.0000, 0.2618, 0.5236, 0.7854, 1.0472, 1.3090, 1.5708, 1.8326, 2.0944,
        2.3562, 2.6180, 2.8798, 3.1416, 3.4034, 3.6652, 3.9270, 4.1888, 4.4506,
        4.7124, 4.9742, 5.2360, 5.4978, 5.7596, 6.0214, 6.2832],
       requires_grad=True)
```



接下来，我们将进行计算，并绘制其输出与其输入之间的关系图：

```
b = torch.sin(a)
plt.plot(a.detach(), b.detach())
```



![autogradyt tutorial](https://docs.pytorch.org/tutorials/_images/sphx_glr_autogradyt_tutorial_001.png)

```
[<matplotlib.lines.Line2D object at 0x7f1e87f69bd0>]
```



让我们仔细看看张量 `b` 。打印出来后，我们可以看到一个指示符，表明它正在跟踪其计算历史：

```
print(b)
```



```
tensor([ 0.0000e+00,  2.5882e-01,  5.0000e-01,  7.0711e-01,  8.6603e-01,
         9.6593e-01,  1.0000e+00,  9.6593e-01,  8.6603e-01,  7.0711e-01,
         5.0000e-01,  2.5882e-01, -8.7423e-08, -2.5882e-01, -5.0000e-01,
        -7.0711e-01, -8.6603e-01, -9.6593e-01, -1.0000e+00, -9.6593e-01,
        -8.6603e-01, -7.0711e-01, -5.0000e-01, -2.5882e-01,  1.7485e-07],
       grad_fn=<SinBackward0>)
```



这个 `grad_fn` 提示我们，在执行反向传播步骤并计算梯度时，我们需要计算该张量所有输入的 sin⁡(x)sin(*x*) 的导数。

让我们再进行一些计算：

```
c = 2 * b
print(c)

d = c + 1
print(d)
```



```
tensor([ 0.0000e+00,  5.1764e-01,  1.0000e+00,  1.4142e+00,  1.7321e+00,
         1.9319e+00,  2.0000e+00,  1.9319e+00,  1.7321e+00,  1.4142e+00,
         1.0000e+00,  5.1764e-01, -1.7485e-07, -5.1764e-01, -1.0000e+00,
        -1.4142e+00, -1.7321e+00, -1.9319e+00, -2.0000e+00, -1.9319e+00,
        -1.7321e+00, -1.4142e+00, -1.0000e+00, -5.1764e-01,  3.4969e-07],
       grad_fn=<MulBackward0>)
tensor([ 1.0000e+00,  1.5176e+00,  2.0000e+00,  2.4142e+00,  2.7321e+00,
         2.9319e+00,  3.0000e+00,  2.9319e+00,  2.7321e+00,  2.4142e+00,
         2.0000e+00,  1.5176e+00,  1.0000e+00,  4.8236e-01, -3.5763e-07,
        -4.1421e-01, -7.3205e-01, -9.3185e-01, -1.0000e+00, -9.3185e-01,
        -7.3205e-01, -4.1421e-01,  4.7684e-07,  4.8236e-01,  1.0000e+00],
       grad_fn=<AddBackward0>)
```



最后，我们来计算一个单元素输出。当你调用 对没有参数的张量使用 `.backward()` 时，它期望调用张量只包含一个元素，就像计算损失函数时一样。

```
out = d.sum()
print(out)
```



```
tensor(25., grad_fn=<SumBackward0>)
```



每个存储在张量中的 `grad_fn` 都允许你使用其 `next_functions` 回溯计算过程，直至其输入。 属性。我们可以在下面看到，深入分析该属性的 `d` 部分。 图中显示了所有先验张量的梯度函数。请注意： `a.grad_fn` 被报告为 `None` ，表明这是函数的一个输入，它本身没有历史记录。

```
print('d:')
print(d.grad_fn)
print(d.grad_fn.next_functions)
print(d.grad_fn.next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
print('\nc:')
print(c.grad_fn)
print('\nb:')
print(b.grad_fn)
print('\na:')
print(a.grad_fn)
```



```
d:
<AddBackward0 object at 0x7f1e8caf86a0>
((<MulBackward0 object at 0x7f1e8caf9e70>, 0), (None, 0))
((<SinBackward0 object at 0x7f1e8caf9e70>, 0), (None, 0))
((<AccumulateGrad object at 0x7f1e8caf86a0>, 0),)
()

c:
<MulBackward0 object at 0x7f1e8caf9e70>

b:
<SinBackward0 object at 0x7f1e8caf9e70>

a:
None
```



有了这些机制，我们如何求导呢？对输出调用 `backward()` 方法，并检查输入…… 使用 `grad` 属性检查渐变：

```
out.backward()
print(a.grad)
plt.plot(a.detach(), a.grad.detach())
```



![autogradyt tutorial](https://docs.pytorch.org/tutorials/_images/sphx_glr_autogradyt_tutorial_002.png)

```
tensor([ 2.0000e+00,  1.9319e+00,  1.7321e+00,  1.4142e+00,  1.0000e+00,
         5.1764e-01, -8.7423e-08, -5.1764e-01, -1.0000e+00, -1.4142e+00,
        -1.7321e+00, -1.9319e+00, -2.0000e+00, -1.9319e+00, -1.7321e+00,
        -1.4142e+00, -1.0000e+00, -5.1764e-01,  2.3850e-08,  5.1764e-01,
         1.0000e+00,  1.4142e+00,  1.7321e+00,  1.9319e+00,  2.0000e+00])

[<matplotlib.lines.Line2D object at 0x7f1ec60aca30>]
```



回顾一下我们得出这个结果所采取的计算步骤：

```
a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
b = torch.sin(a)
c = 2 * b
d = c + 1
out = d.sum()
```



像我们计算 `d` 那样加上一个常数，并不会改变导数。这样就剩下 c=2∗b=2∗sin⁡(a)*c*=2∗*b*=2∗sin(*a*) ，它的导数应该是 2∗cos⁡(a)2∗cos(*a*) 。观察上面的图，我们看到的正是如此。

请注意，只有计算中的*叶节点*才会计算梯度。例如，如果您尝试 `print(c.grad)` 则会返回空值。 `None` 。在这个简单的例子中，只有输入是叶节点，因此只有它计算了梯度。

## 自动毕业培训#

我们已经简要了解了自动微分的工作原理，但它在实际应用中究竟如何呢？让我们定义一个小型模型，并观察它在一次训练批次后会发生哪些变化。首先，定义一些常量、我们的模型以及一些输入和输出的替代值：

```
BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.layer1 = torch.nn.Linear(DIM_IN, HIDDEN_SIZE)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(HIDDEN_SIZE, DIM_OUT)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

model = TinyModel()
```



您可能会注意到，我们从未明确指出这一点。 模型的层需要 `requires_grad=True` 。在子类中 `torch.nn.Module` ，假设我们想要跟踪层权重上的梯度以进行学习。

如果我们查看模型的各个层，我们可以检查权重值，并验证是否尚未计算任何梯度：

```
print(model.layer2.weight[0][0:10]) # just a small slice
print(model.layer2.weight.grad)
```



```
tensor([-0.0808, -0.0980, -0.0697,  0.0915,  0.0925, -0.0457,  0.0614,  0.0523,
        -0.0804, -0.0248], grad_fn=<SliceBackward0>)
None
```



让我们看看运行一个训练批次后情况会有什么变化。对于损失函数，我们将使用 `prediction` 与 `ideal_output` 之间欧氏距离的平方，并使用基本的随机梯度下降优化器。

```
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

prediction = model(some_input)

loss = (ideal_output - prediction).pow(2).sum()
print(loss)
```



```
tensor(145.1727, grad_fn=<SumBackward0>)
```



现在，我们调用 `loss.backward()` 函数，看看会发生什么：

```
loss.backward()
print(model.layer2.weight[0][0:10])
print(model.layer2.weight.grad[0][0:10])
```



```
tensor([-0.0808, -0.0980, -0.0697,  0.0915,  0.0925, -0.0457,  0.0614,  0.0523,
        -0.0804, -0.0248], grad_fn=<SliceBackward0>)
tensor([-1.5426e+00,  1.9859e+00, -1.5431e-01,  1.1749e+00, -2.2813e+00,
        -4.7378e+00,  1.0719e-03,  1.3172e+00,  1.7830e+00, -1.0519e+00])
```



我们可以看到，每个学习权重的梯度都已经计算出来了，但权重本身保持不变，因为我们还没有运行优化器。优化器负责根据计算出的梯度更新模型权重。

```
optimizer.step()
print(model.layer2.weight[0][0:10])
print(model.layer2.weight.grad[0][0:10])
```



```
tensor([-0.0792, -0.1000, -0.0695,  0.0904,  0.0948, -0.0409,  0.0614,  0.0510,
        -0.0822, -0.0237], grad_fn=<SliceBackward0>)
tensor([-1.5426e+00,  1.9859e+00, -1.5431e-01,  1.1749e+00, -2.2813e+00,
        -4.7378e+00,  1.0719e-03,  1.3172e+00,  1.7830e+00, -1.0519e+00])
```



你应该可以看到 `layer2` 的权重发生了变化。

关于这个过程，有一点很重要：打电话之后…… `optimizer.step()` ，需要调用 `optimizer.zero_grad()` ，否则每次运行 `loss.backward()` 时，学习权重上的梯度都会累积：

```
print(model.layer2.weight.grad[0][0:10])

for i in range(0, 5):
    prediction = model(some_input)
    loss = (ideal_output - prediction).pow(2).sum()
    loss.backward()

print(model.layer2.weight.grad[0][0:10])

optimizer.zero_grad(set_to_none=False)

print(model.layer2.weight.grad[0][0:10])
```



```
tensor([-1.5426e+00,  1.9859e+00, -1.5431e-01,  1.1749e+00, -2.2813e+00,
        -4.7378e+00,  1.0719e-03,  1.3172e+00,  1.7830e+00, -1.0519e+00])
tensor([  4.1569,  17.4602,  -7.8189,  -4.2132, -14.6893, -19.3380,  -2.3023,
          1.5380,  18.6185,  -9.2201])
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
```



运行上述单元格后，您应该会看到运行后的结果。 多次调用 `loss.backward()` 会导致大部分梯度的幅值过大。如果在运行下一个训练批次之前没有将梯度归零，就会导致梯度以这种方式膨胀，从而造成错误且不可预测的学习结果。

## 关闭和开启自动渐变#

在某些情况下，您需要对是否启用自动微分进行精细控制。根据具体情况，有多种方法可以实现这一点。

最简单的方法是直接修改张量的 `requires_grad` 标志：

```
a = torch.ones(2, 3, requires_grad=True)
print(a)

b1 = 2 * a
print(b1)

a.requires_grad = False
b2 = 2 * a
print(b2)
```



```
tensor([[1., 1., 1.],
        [1., 1., 1.]], requires_grad=True)
tensor([[2., 2., 2.],
        [2., 2., 2.]], grad_fn=<MulBackward0>)
tensor([[2., 2., 2.],
        [2., 2., 2.]])
```



在上面的单元格中，我们看到 `b1` 具有 `grad_fn` （即跟踪计算历史），这符合预期，因为它源自一个启用了自动微分的张量 `a` 。当我们显式地使用 `a.requires_grad = False` 关闭自动微分时，计算历史将不再被跟踪，正如我们在计算 `b2` 时所看到的那样。

如果您只需要暂时关闭自动微分功能，更好的方法是使用 `torch.no_grad()` ：

```
a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3

c1 = a + b
print(c1)

with torch.no_grad():
    c2 = a + b

print(c2)

c3 = a * b
print(c3)
```



```
tensor([[5., 5., 5.],
        [5., 5., 5.]], grad_fn=<AddBackward0>)
tensor([[5., 5., 5.],
        [5., 5., 5.]])
tensor([[6., 6., 6.],
        [6., 6., 6.]], grad_fn=<MulBackward0>)
```



`torch.no_grad()` 也可以用作函数或方法装饰器：

```
def add_tensors1(x, y):
    return x + y

@torch.no_grad()
def add_tensors2(x, y):
    return x + y


a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3

c1 = add_tensors1(a, b)
print(c1)

c2 = add_tensors2(a, b)
print(c2)
```



```
tensor([[5., 5., 5.],
        [5., 5., 5.]], grad_fn=<AddBackward0>)
tensor([[5., 5., 5.],
        [5., 5., 5.]])
```



还有一个对应的上下文管理器 `torch.enable_grad()` ，用于在自动微调功能尚未启用时启用它。它也可以用作装饰器。

最后，您可能有一个需要梯度跟踪的张量，但您想要一个不需要梯度跟踪的副本。为此，我们可以使用 `Tensor` 对象。 `detach()` 方法——它会创建一个*被分离的*张量的副本。 根据计算历史记录：

```
x = torch.rand(5, requires_grad=True)
y = x.detach()

print(x)
print(y)
```



```
tensor([0.5345, 0.1485, 0.1025, 0.4849, 0.2943], requires_grad=True)
tensor([0.5345, 0.1485, 0.1025, 0.4849, 0.2943])
```



我们之前为了绘制一些张量图而做了这件事。这是因为 `matplotlib` 需要 NumPy 数组作为输入，而对于 requires_grad=True 的张量，从 PyTorch 张量到 NumPy 数组的隐式转换并未启用。创建一个分离的副本可以让我们继续进行后续操作。

### 自动升压和就地作业#

到目前为止，本笔记本中的每个示例都使用了变量来捕获计算的中间值。自动微分 (Autograd) 需要这些中间值来执行梯度计算。 *因此，在使用自动微分时，必须谨慎地进行原地操作。* 这样做可能会破坏在 `backward()` 调用中计算导数所需的信息。如果您尝试对需要自动微分的叶子变量进行原地操作，PyTorch 甚至会阻止您的操作，如下所示。

笔记

> 以下代码单元格会抛出运行时错误。这是预期行为。

```
a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
torch.sin_(a)
```



## 自学毕业生简介#

Autograd 会详细跟踪计算的每一步。这样的计算历史记录，结合时间信息，可以构成一个便捷的性能分析器——而 Autograd 已经内置了这项功能。以下是一个简单的使用示例：

```
device = torch.device('cpu')
run_on_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda')
    run_on_gpu = True

x = torch.randn(2, 3, requires_grad=True)
y = torch.rand(2, 3, requires_grad=True)
z = torch.ones(2, 3, requires_grad=True)

with torch.autograd.profiler.profile(use_cuda=run_on_gpu) as prf:
    for _ in range(1000):
        z = (z / x) * y

print(prf.key_averages().table(sort_by='self_cpu_time_total'))
```



```
/var/lib/workspace/beginner_source/introyt/autogradyt_tutorial.py:485: FutureWarning:

The attribute `use_cuda` will be deprecated soon, please use ``use_device = 'cuda'`` instead.

------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                     cudaEventRecord        42.34%       9.120ms        42.34%       9.120ms       2.280us       0.000us         0.00%       0.000us       0.000us          4000
                           aten::div        21.22%       4.570ms        21.22%       4.570ms       4.570us       8.458ms        50.49%       8.458ms       8.458us          1000
                           aten::mul        20.41%       4.396ms        20.41%       4.396ms       4.396us       8.294ms        49.51%       8.294ms       8.294us          1000
          cudaGetDeviceProperties_v2        15.92%       3.429ms        15.92%       3.429ms       1.715ms       0.000us         0.00%       0.000us       0.000us             2
               cudaDeviceSynchronize         0.08%      16.870us         0.08%      16.870us      16.870us       0.000us         0.00%       0.000us       0.000us             1
    cudaDeviceGetStreamPriorityRange         0.01%       3.070us         0.01%       3.070us       3.070us       0.000us         0.00%       0.000us       0.000us             1
               cudaStreamIsCapturing         0.01%       2.770us         0.01%       2.770us       0.923us       0.000us         0.00%       0.000us       0.000us             3
                  cudaGetDeviceCount         0.00%       0.360us         0.00%       0.360us       0.180us       0.000us         0.00%       0.000us       0.000us             2
------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 21.539ms
Self CUDA time total: 16.752ms
```



分析器还可以标记代码的各个子块，并进行细分。 根据输入张量形状提取数据，并将数据导出为 Chrome 跟踪工具。 文件。有关 API 的完整详细信息，请参阅 [文档 ](https://pytorch.org/docs/stable/autograd.html#profiler)。

## 进阶主题：更多自动微分细节和高级 API#

如果你有一个函数，它的输入是 n 维的，输出是 m 维的 输出 y⃗=f(x⃗)*y*=*f*(*x*) ，完整的梯度是一个矩阵 每个输出对每个输入的导数，称为 *雅可比矩阵：*

J=(∂y1∂x1⋯∂y1∂xn⋮⋱⋮∂ym∂x1⋯∂ym∂xn)*J*=∂*x*1∂*y*1⋮∂*x*1∂*y**m*⋯⋱⋯∂*x**n*∂*y*1⋮∂*x**n*∂*y**m*

如果你有第二个函数 l=g(y⃗)*l*=*g*(*y*) ，它接受 m 维输入（即与上面的输出维度相同），并返回标量输出，你可以将它相对于 y⃗*y* 的梯度表示为一个列向量， v=(∂l∂y1⋯∂l∂ym)T*v*=(∂*y*1∂*l*⋯∂*y**m*∂*l*)*T* ——实际上就是一个单列雅可比矩阵。

更具体地说，可以将第一个函数想象成你的 PyTorch 模型（可能有很多输入和很多输出），将第二个函数想象成损失函数（以模型的输出作为输入，损失值作为标量输出）。

如果我们将第一个函数的雅可比矩阵乘以第二个函数的梯度，并应用链式法则，我们得到：

JT⋅v=(∂y1∂x1⋯∂ym∂x1⋮⋱⋮∂y1∂xn⋯∂ym∂xn)(∂l∂y1⋮∂l∂ym)=(∂l∂x1⋮∂l∂xn)*J**T*⋅*v*=∂*x*1∂*y*1⋮∂*x**n*∂*y*1⋯⋱⋯∂*x*1∂*y**m*⋮∂*x**n*∂*y**m*∂*y*1∂*l*⋮∂*y**m*∂*l*=∂*x*1∂*l*⋮∂*x**n*∂*l*

注意：您也可以使用等效操作 vT⋅J*v**T*⋅*J* ，并得到一个行向量。

得到的列向量是*第二个函数相对于第一个函数的输入的梯度* ——或者，就我们的模型和损失函数而言，是损失函数相对于模型输入的梯度。

**``torch.autograd`` 是一个用于计算这些乘积的引擎。** 这就是我们在反向传播过程中累积学习权重梯度的方法。

因此，` `backward()` 函数*也*可以接受一个可选的向量输入。该向量表示张量的梯度集合，这些梯度将乘以前一个经过自动微分追踪的张量的雅可比矩阵。让我们用一个小的向量来举个具体的例子：

```
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
```



```
tensor([ 197.8760, 1628.7651, -605.4437], grad_fn=<MulBackward0>)
```



如果我们现在尝试调用 `y.backward()` ，将会得到一个运行时错误，并提示梯度只能对标量输出进行 *隐式*计算。对于多维输出，自动微分函数需要我们提供三个输出的梯度，以便将其乘以雅可比矩阵：

```
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float) # stand-in for gradients
y.backward(v)

print(x.grad)
```



```
tensor([1.0240e+02, 1.0240e+03, 1.0240e-01])
```



（请注意，输出梯度都与 2 的幂有关——这正是我们预期在重复倍增操作下的结果。）

### 高级 API#

autograd 提供了一个 API，可以直接访问重要的微分矩阵和向量运算。特别是，它允许您针对特定输入计算特定函数的雅可比矩阵和 *Hessian* 矩阵。（Hessian 矩阵类似于雅可比矩阵，但它表示所有 *二阶*偏导数。）它还提供了与这些矩阵进行向量积运算的方法。

让我们计算一个简单函数的雅可比矩阵，该函数有两个单元素输入：

```
def exp_adder(x, y):
    return 2 * x.exp() + 3 * y

inputs = (torch.rand(1), torch.rand(1)) # arguments for the function
print(inputs)
torch.autograd.functional.jacobian(exp_adder, inputs)
```



```
(tensor([0.9773]), tensor([0.8007]))

(tensor([[5.3145]]), tensor([[3.]]))
```



仔细观察，第一个输出应该等于 2ex2*e**x* （因为 ex*e**x* 的导数是 ex*e**x* ），第二个值应该是 3。

当然，你也可以使用更高阶的张量来实现这一点：

```
inputs = (torch.rand(3), torch.rand(3)) # arguments for the function
print(inputs)
torch.autograd.functional.jacobian(exp_adder, inputs)
```



```
(tensor([0.5580, 0.7055, 0.4242]), tensor([0.6580, 0.4605, 0.3707]))

(tensor([[3.4942, 0.0000, 0.0000],
        [0.0000, 4.0498, 0.0000],
        [0.0000, 0.0000, 3.0566]]), tensor([[3., 0., 0.],
        [0., 3., 0.],
        [0., 0., 3.]]))
```



`torch.autograd.functional.hessian()` 方法的工作方式相同（假设你的函数是二阶可微的），但返回所有二阶导数的矩阵。

如果提供向量，还可以使用函数直接计算向量雅可比积：

```
def do_some_doubling(x):
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    return y

inputs = torch.randn(3)
my_gradients = torch.tensor([0.1, 1.0, 0.0001])
torch.autograd.functional.vjp(do_some_doubling, inputs, v=my_gradients)
```



```
(tensor([  358.2216,  1080.0546, -1546.7737]), tensor([1.0240e+02, 1.0240e+03, 1.0240e-01]))
```



`torch.autograd.functional.jvp()` 方法执行与 `vjp()` 相同的矩阵乘法，只是操作数顺序相反。vhp `vhp()` `hvp()` 方法对向量 Hessian 积执行相同的操作。

有关更多信息，包括性能说明 [请参阅函数式 API 的文档。](https://pytorch.org/docs/stable/autograd.html#functional-higher-level-api)