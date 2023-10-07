# coding: utf-8
import sys, os

import numpy as np
# import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定

# 读入数据
# Load the data
import pandas as pd
train = pd.read_csv("./digit-recognizer/train.csv")
test = pd.read_csv("./digit-recognizer/test.csv")

Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

# Normalize the data
X_train = X_train / 255.0
test = test / 255.0

X_train = X_train.values.reshape(-1, 784)
test = test.values.reshape(-1, 784)
Y_train = pd.get_dummies(Y_train)
Y_train = Y_train.values

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# Above is the first step: set the weights for linear layers.

iters_num = 60  # 适当设定循环的次数
train_size = X_train.shape[0]
batch_size = 300
learning_rate = 0.5

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size*1000, 1)

for i in range(iters_num):
    print(i)
    batch_mask = np.random.choice(train_size, batch_size) # randomly choose a batch from x
    x_batch = X_train[batch_mask]
    t_batch = Y_train[batch_mask]
    
    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch)
    
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        # network.params[key] = network.params[key] - learning_rate * grad[key]
        network.params[key] -= learning_rate * grad[key]

    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    train_acc = network.accuracy(X_train, Y_train)
    # test_acc = network.accuracy(x_test, t_test)
    test_acc = 0
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc),'loss:',loss)

# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()