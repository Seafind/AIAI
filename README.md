
# 人工智能的数学思维 作业

## 测试单隐藏层神经网络的拟合能力

**张荣昊 10245101445**

### 实验目的

探究隐藏层神经元数量和不同激活函数（S 型函数与 ReLU 函数）对神经网络拟合能力

### 实验设置

#### 目标函数

生成 1000 个均匀采样点作为训练数据。

#### 单隐藏层神经网络参数

神经元数量：`[1, 5, 10, 100, 500, 1000]`

激活函数：tansig（S 型）与 poslin（ReLU）

### 实验代码

#### S 型函数

```matlab
clear; clc; close all;

% 生成训练数据--拟合的目标函数
x = linspace(0, 1, 1000);
y = power(sin(2*pi*x),3);

% 隐藏层的不同神经元数
neurons_list = [1,5,10,100,500,1000];
colors = {'r','g','b','m','c','k'};

% 使用 S 型激活函数
y_tansig = tansig(x);    % S 型激活函数

% 绘图
figure;
for i = 1:length(neurons_list)
    % 创建前馈神经网络
    net = feedforwardnet(neurons_list(i)); 

    % 方案 1：激活函数为 S 型函数（即 tansig
    net.layers{1}.transferFcn = 'tansig';  % TanSig
    net.layers{2}.transferFcn = 'purelin'; % 输出层保持线性 

    % 配置网络
    net = configure(net, x, y);

    % 设置最小显示信息
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;

    % 训练神经网络
    net = train(net, x, y);

    % 预测输出
    y_pred = net(x);

    % 绘图
    subplot(2,3,i);
    plot(x, y, 'k--', 'LineWidth', 1.5); hold on;
    plot(x, y_pred, colors{i}, 'LineWidth', 2);
    title([num2str(neurons_list(i)) ' Hidden Neurons']);
    xlabel('x'); ylabel('f(x)');
    legend('True Function', 'NN Approximation');
    grid on;
end

sgtitle('Single Hidden Layer Neural Network Approximation of sin(2πx)^3');
```

#### ReLU 函数

```matlab
clear; clc; close all;

% 生成训练数据--拟合的目标函数
x = linspace(0, 1, 1000);
y = power(sin(2*pi*x),3);

% 隐藏层的不同神经元数
neurons_list = [1,5,10,100,500,1000];
colors = {'r','g','b','m','c','k'};

% 使用 ReLU 函数
y_relu = poslin(x);      % ReLU 函数：max(0, x)

% 绘图
figure;
for i = 1:length(neurons_list)
    % 创建前馈神经网络
    net = feedforwardnet(neurons_list(i)); 

    % 方案 2：激活函数为 ReLU（即 poslin）
    net.layers{1}.transferFcn = 'poslin';  % ReLU
    net.layers{2}.transferFcn = 'purelin'; % 输出层保持线性

    % 配置网络
    net = configure(net, x, y);

    % 设置最小显示信息
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;

    % 训练神经网络
    net = train(net, x, y);

    % 预测输出
    y_pred = net(x);

    % 绘图
    subplot(2,3,i);
    plot(x, y, 'k--', 'LineWidth', 1.5); hold on;
    plot(x, y_pred, colors{i}, 'LineWidth', 2);
    title([num2str(neurons_list(i)) ' Hidden Neurons']);
    xlabel('x'); ylabel('f(x)');
    legend('True Function', 'NN Approximation');
    grid on;
end

sgtitle('Single Hidden Layer Neural Network Approximation of sin(2πx)');
```

### 实验结果

#### S 型函数

![image](https://github.com/user-attachments/assets/c89573d3-f9b7-40cb-95b7-50393ed64197)

#### ReLU 函数

![image](https://github.com/user-attachments/assets/f516abb6-1b5c-485f-9e83-6d17b37648c4)

### 结果分析

从实验结果中可以看到，随着隐藏层神经元数量的增加，S 型函数的拟合效果在开始时有明显的提升，但是当隐藏层神经元数量达到 500 时，S 型函数 tansig 出现了少量的 “毛刺”，而随着隐藏层神经元数量继续增加，达到 1000 时，整个图像布满 “毛刺”。

ReLU 函数则不同，开始时拟合效果比 S 型函数差很多，但是隐藏层神经元数量越多拟合效果越好，而且没有 “毛刺” 现象。

这与所学知识相符合，实验成功。
```
