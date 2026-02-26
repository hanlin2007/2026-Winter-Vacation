
> [!搭建工具链：]
> Anaconda+Pytorch（CUDA支持版本）+**VSCode**+**Pycharm**+Jupyter+GithubSync

结论：并不是Anaconda的配置和环境问题，而是Pycharm旧版本在连接Conda环境的时候出现了配置设置文件和缓存的问题，使用VSCode就可以直接解决，最终彻底卸载并更新Pycharm版本2025.3.2.1，也解决了所有问题。

==以下日志记录了我环境配置过程中所有的操作步骤和遇到的问题及解决方案==
## 第一步，下载并安装Anaconda

Anaconda是一个python环境的管理工具，可以使用Anaconda来创建多个python的版本支持环境，用于快速在不同项目中配置开发环境。

Anaconda下载之后有一个Register为Default Python Interpreter，下面写了一个recommend，一开始选了之后非常担心出了问题，后来发现其实没有任何影响。

听了一个视频的介绍，就选择了C盘，全部按照了默认设置引导进行安装，后续感觉其实可以放在D盘的，但是很多路径设置，比如用户和系统环境变量，再比如Pycharm和VSCode中Conda解释器的路径都已经写好了，就不改了

## 第二步，设置python环境，下载pytorch

可以打开anaconda navigator 利用图形化+命令窗口进行环境配置
Open Terminal 
pip3 install pytorch torchvision torchaudio
复制命令给ai添加清华源地址 可以加速下载

## 第三步，将python环境导入pycharm，运行测试文件

按照前面的要求安装anaconda，但仍然没有解决Pycharm的导入扫描问题，后续补充了Anaconda的Path环境变量，并且在Path to Conda中手动选择了C:\Users\lihanlin2007\anaconda3\envs\pytorch_learning_2026\python.exe

应该是在这里**解决了导入扫描问题**。但Pycharm仍然提示
C:\Users\lihanlin2007\anaconda3\envs\pytorch_learning_2026\python.exe: can't open file 'D:\\Pycharm 2025.2.4\\info': No such file or directory

这个错误 `can't open file 'D:\\Pycharm 2025.2.4\\info'`表明 PyCharm 在**尝试读取自身配置文件时**，文件路径错误或文件丢失。这与你选择的 `python.exe`路径无关，而是 PyCharm 内部的配置问题。
这很可能是因为我在之前导入扫描错误那里，对pycharm的配置文件进行了改名备份有关

**所以我最终尝试解决方案：彻底删除Pycharm以及所有配置文件**
点此电脑（管理员身份）打开C盘用户文件夹中隐藏的项目APPDATA保存了Local、Roaming中Jetbrain的命名配置文件，全部删除，再删除IDE，从官网重新下载，终于彻底删除了Pycharm
重新打开后跳过 Import Settings 直接创建项目，选择已存在的Conda环境，就终于成功扫描到了 期间还因为Pycharm太大了两次让电脑直接强制重启

任务管理器快捷键改成了Ctrl+Alt+T（Task）看到CPU占比接近50%！！

---
## Anaconda & VSCode & GithubSync的使用细节

- **Anaconda**

在开始，全部里面打开Anaconda Prompt
1.`conda -V `   
用来检测环境变量
2.`conda create -n pytorch(创建的环境的名字) python=3.9`  
创建一个新的python环境，这一步创建，跟在Anaconda Navigator中初始化创建一样
`Proceed?([y]/n)` 会提示接下来的操作选项，并且指出推荐的选项
3.`conda activate pytorch`  激活环境，从base变成名称为pytorch的环境

接下来在这个环境中安装pytorch框架/相关包和模块
1.`nvidia-smi`用于查看显卡状态，会显示CUDA版本
2.`pip install...`打开Pytorch官网，选择对应版本的Pytorch框架
注意，只能选择比自己CUDA版本小的，安装13.0失败，12.8就成功了
3.终端输入python进入了python编辑器界面 exit( ) 退出
4.装包速度慢，用清华源来换pip的下载源
`pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`

- **VSCode**

在VSCode中安装python解释器以及Jupyter，打开新建的文件夹，创建测试文件
在右下角直接选择conda创建的pytorch环境！！太快了 问题直接解决了

如果右下角找不到conda创建的pytorch环境怎么办？在Anaconda Prompt中，进入pytorch环境（按前面的激活），输入where python，将带有pytorch的路径直接复制到interpreter，即为正确的解释器路径

对于JupyterNotebook的测试，需要额外选择一个内核即可，初次运行没有内核直接按照指示安装即可

- **GithubSync**

VSCode里面好像自己就连上源代码管理了我也不知道为啥，选择连接到远程仓库，在菜单里选择以Github方式继续，找到自己创建好的仓库。

git remote -v 会显示已配置的远程仓库
不能直接提交修改，如果先创建仓库，会存在本地不存在的Readme

拉取远程内容（使用你已有的远程仓库名）
git pull Pytorch_Learning_2026 main --allow-unrelated-histories

如果成功，推送到远程，注意推送过程要把梯子关掉
git push Pytorch_Learning_2026 main

手动取消git代理，自己也关闭代理
git config --global --unset http.proxy
git config --global --unset https.proxy

只上传了文件夹和一些默认文件，没有上传自己新增的文件？
添加所有文件
git add .

git commit -m "添加 test.py 文件"

git push Pytorch_Learning_2026 main