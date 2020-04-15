---
layout:     post                    # 使用的布局（不需要改）
title:      Linear Regression               # 标题 
subtitle:   assignment of Machine Learning course  #副标题
date:       2019-11-09              # 时间
author:     Zongyou Liu             # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - work
---
## 实验过程  
在线性回归中分别使用BGD和SGD的方法进行迭代，训练过程中使用BGD和SGD对应的损失函数值如图1所示，训练后对测试集的测试结果与真实值的比较如图3所示（图3对X和Y进行了标准化） 
![linear1](https://github.com/BuleSky233/BuleSky233.github.io/raw/master/img/linear1.png)

### 下讨论BGD和SGD的区别：
1. 由图1可看出，SGD比BGD更快达到一个稳定的较小值，即SGD达到收敛经过的迭代次数少于BGD。实际上这从算法的本质也可看出，SGD是每次算完一个样本就更改一次θ参数，而BGD是算完所有的样本才同时更改全部θ，所以SGD会更快收敛。 
2. 图3的BGD和SGD预测的结果几乎重合，可见对于本数据集来说，这两种方法的最终效果是一样的。在训练时BGD达到的最小损失函数值分别为62.61139175787707，SGD达到的最小损失函数值为 62.61610565278707，BGD的效果略好于SGD。实际上，对于某些数据集应用线性回归，SGD可能只能找到局部最优解，而BGD一定能找到全局最优解。 
所以，对于样本数不多的数据集，最好用BGD方法迭代，而对于样本数很多的数据集，BGD的迭代速度可能会非常慢，此时SGD也是一个很好的选择。 

## 源代码
```
# coding=utf-8
import random    
from sklearn import preprocessing    
import numpy as np    
import pandas as pd    
import matplotlib.pyplot as plt    
from sklearn.preprocessing import StandardScaler

def computeCost(X, Y, theta):
    p = np.dot(((np.dot(X.T, theta) - Y)).T, np.dot(X.T, theta) - Y)
    return (1 / 2) * (p[0][0])

def BGD(X, Y, theta, rate=0.00005, thredsome=0.1, maxstep=10000):
    # update theta
    cost = computeCost(X, Y, theta)
    # print(cost)
    picturelist = []
    step = 0
    while cost > thredsome and step < maxstep:
        tem = theta - rate * np.dot(X, np.dot(X.T, theta) - Y)
        theta = tem
        cost = computeCost(X, Y, theta)
        # print(cost)
        picturelist.append(cost)
        step += 1
    # if cost>thredsome:
    #     print("发散")
    return theta, step, picturelist

def SGD(X, Y, theta, rate=0.0001, thredsome=0.1, maxstep=10000):

    picturelist = []
    step = 0
    cost = computeCost(X, Y, theta)
    picturelist.append(cost)
    dimension, samplenum = X.shape
    while cost > thredsome and step < maxstep:
        for i in range(samplenum):
            predict_y = np.dot(X[:, i], theta)[0]
            for j in range(dimension):
                theta[j] -= rate * (predict_y - Y[i][0]) * X[j][i]
        cost = computeCost(X, Y, theta)
        picturelist.append(cost)
        step += 1
    # if cost>thredsome:
    #     print("发散")
    return theta, step, picturelist

f = open('housing.data')
df = pd.read_csv(f, header=None)
# feature=[]
# for i in df:
#     y=i.split()
#     for j in y:
#         j=float(j)
#     feature.append(y)
# print(feature)
df = df.values.tolist()

featurelist = []
for i in df:
    y = i[0].split()
    for j in range(len(y)):
        y[j] = float(y[j])
    featurelist.append(y)

X = []
Y = []

for i in featurelist:
    tem = []
    tem.append(i[-1])
    Y.append(tem)
    X.append(i[0:-1])

Y = np.array(Y)
X = np.array(X)


stand = StandardScaler()
X = stand.fit_transform(X)
Y = stand.fit_transform(Y)
# cur_thete_SGD,lost_collection=sgd(X,Y)
X = X.T
X = np.insert(X, 0, [1], axis=0)
train_X=X[:,0:455]
test_X=X[:,455:]
print(train_X.shape)
print(test_X.shape)
train_Y=Y[0:455]
test_Y=Y[455:]
print(train_Y.shape)
print(test_Y.shape)
#index_max = np.argmax(X, axis=1)
# index_min = np.argmin(X,axis=1)
# for i in range(len(X)):
#     if i!=0:
#         for j in range(len(X[i])):
#             X[i][j]=(X[i][j]-X[i][index_min[i]])/(X[i][index_max[i]]-X[i][index_min[i]])
theta = np.ones([len(X), 1], dtype=float)

cur_theta_BGD, step, lost_BGD = BGD(train_X, train_Y, theta)
cur_thete_SGD,step, lost_SGD=SGD(train_X,train_Y,theta)

print("BGD的最小损失函数值为",lost_BGD[-1])
print("SGD的最小损失函数值为",lost_SGD[-1])

picturex_BGD=np.arange(len(lost_BGD))
picturex_SGD=np.arange(len(lost_SGD))
plt.plot(picturex_BGD,lost_BGD,color='blue', label='BGD')
plt.plot(picturex_SGD,lost_SGD,color='red', label='SGD')
plt.legend(["BGD","SGD"])
plt.title(" the traning loss function of BGD and SGD")
plt.savefig("./the traning loss function of BGD and SGD 2.png")
plt.show()

predict_BGD=np.dot(test_X.T,cur_theta_BGD)
predict_SGD=np.dot(test_X.T,cur_thete_SGD)
plt.plot(np.arange(len(test_X[0])),predict_BGD,color='blue', label='predict_BGD')
plt.plot(np.arange(len(test_X[0])),predict_SGD,color='red', label='predict_SGD')
plt.plot(np.arange(len(test_X[0])),test_Y,color='yellow',label='real')
plt.legend(["predict_BGD","predict_SGD","real"])
plt.title(" test housing price")
plt.savefig("./test housing price 2.png")
plt.show()
```
