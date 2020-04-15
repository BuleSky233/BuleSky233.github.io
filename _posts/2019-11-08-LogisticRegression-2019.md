---
layout:     post                    # 使用的布局（不需要改）
title:      Project2:Logistic Regression               # 标题 
subtitle:   assignment of machine learning course #副标题
date:       2019-11-08             # 时间
author:     Zongyou Liu                      # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - project 
    - machine learning
---

## 问题背景:
日常生活中，我们可能会收到大量邮件，而其中有一些是垃圾邮件，比如广告或者是欺诈邮件。为了给用户更好的体验，垃圾
邮件的识别就显得尤为重要。如果能准确识别出垃圾邮件并进行拦截，则用户收到的就都是非垃圾邮件，可以使得他们检查邮
箱更加方便快捷，免受从一大堆邮件中找出有用邮件的烦恼。垃圾邮件的检测属于二分类问题，机器学习算法可以很好的解决这类问题，下面使用logistic 回归来进行邮件的二分类。
## 方法与步骤:
1.数据预处理，对数据进行标准化，按3：1的比例划分出训练集和测试集。该数据有 4601个样本，其中正样本（垃圾邮件）
1813 个，负样本2788个，每个样本有 57个特征。

2.使用 logistic 函数 1/（1+e^(-θ^T*X)）作为训练模型，对训练集使用梯度上升法进行 10000次迭代后，训练出的参数θ（总
共 58 个，第一个为常数项）为
[-5.66522636，- 0.06828972，-0.20628317，0.10546832，3.10297088，0.36106111，0.22699143，0.86939079，0.23966098，0.1606796 ，0.03151361，-0.08944871，-0.11330708，
-0.03191618，0.01821621，0.21057288，0.77735843，0.28912783，0.06513675，0.12167821，0.53700456，0.29366183，0.30918639，0.93801747，0.13717517，- 3.01317382，-
0.81718885，- 15.69417749，0.24736742，- 1.73109609，- 0.19860554，0.08021923，0.24150568，- 0.39586339，0.09765889，- 0.88495734，0.42288701，- 0.02595522，- 0.1280041 ，
-0.43477088，-0.09529861， -5.77358322，- 1.88129429，-0.18538062，-1.0299963 ，-0.74994012，-1.19881205，- 0.19024502，-1.04144982，-0.37731927，-0.12685026，-
0.03197163，0.20113023，1.20345509，0.93164543，- 0.22929524，1.24627483，0.49145903]
训练过程中的似然函数值的变化情况如下图所示，可见似然函数的值在训练过程中不断升高,最终达到收敛 

![logistic1](https://raw.githubusercontent.com/BuleSky233/BuleSky233.github.io/master/img/logistic1.png)  
3.使用训练出的模型对测试集进行测试，检验模型的正确性
## 实验结果：
在测试集中的预测邮件是否是垃圾邮件的正确率达到了 0.94，具体结果如下图所示
![logistic2](https://raw.githubusercontent.com/BuleSky233/BuleSky233.github.io/master/img/logistic2.png)
## 对比实验：
调用 sklearn 包里的库函数 LogisticRegression 进行二分类，准确率为 0.93（如下图所示）。可见，我自己编写的 logistic回归
函数准确率更高，在此数据集下的分类结果要优于 sklearn 包里的库函数。
![logistic3](https://raw.githubusercontent.com/BuleSky233/BuleSky233.github.io/master/img/logistic3.png)
## 结果讨论：
从该数据集上看，利用 logistic 回归进行二分类问题效果很不错，正确率达到了 0.94。且 logistic 回归非常容易实现，使用梯度
上升法极大化似然函数的求导结果与线性回归是格式统一的，只需在线性回归的代码基础上修改预测函数 h(x)即可。但是
logistic 回归也有一些缺点，比如因为它本质上是一个线性的分类器，所以处理不好特征之间相关的情况。且当特征空间很大时，
logistic 回归的收敛速度会很慢，性能不高。
## 源代码：
代码源文件在我的github中
```
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def logist(z):
 ans = []
 for i in z:
 tem = []
 tem.append(1 / (1 + math.exp(-i[0])))
 ans.append(tem)
 return ans
def computeCost(X, Y, theta):
 p = np.dot((logist(np.dot(X.T, theta)) - Y).T, logist(np.dot(X.T,
theta)) - Y)
 return (1 / 2) * (p[0][0])
def logistic_regression(X, Y, theta, rate=0.001, thredsome=0.1,
maxstep=10000):
 # update theta
 cost = computeCost(X, Y, theta)
 picturelist = []
 step = 0
 while cost > thredsome and step < maxstep:
 tem = theta - rate * np.dot(X, logist(np.dot(X.T, theta)) - Y)
 theta = tem
 cost = computeCost(X, Y, theta)
 picturelist.append(cost)
 step += 1
 # if cost>thredsome:
 # print("发散")
 return theta, step, picturelist
f = open('spambase.data')
df = pd.read_csv(f, header=None)
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
X = X.T
X = np.insert(X, 0, [1], axis=0)
# train_X=X[:,0:4000]
# test_X=X[:,4000:]
train_X, test_X, train_Y, test_Y = train_test_split(X.T, Y)
train_X = train_X.T
test_X = test_X.T
print(train_X.shape)
print(test_X.shape)
print(train_Y.shape)
print(test_Y.shape)
theta = np.zeros([len(X), 1], dtype=float)
cur_theta_BGD, step, lost_BGD = logistic_regression(train_X, train_Y, theta)
print(cur_theta_BGD)
predict_BGD = logist(np.dot(test_X.T, cur_theta_BGD))
predict_Y = []
#预测正确的邮件数
predict_spam = 0
predict_good = 0
#测试集里的邮件数
test_good = 0
test_spam = 0
for i in test_Y:
 if i[0] == 1:
 test_spam += 1
 else:
 test_good += 1
for i in predict_BGD:
 if i[0] >= 0.5:
 predict_Y.append(1)
  else:
 predict_Y.append(0)
for i in range(len(predict_Y)):
 if predict_Y[i] == test_Y[i][0]:
 if predict_Y[i] == 1:
 predict_spam += 1
 else:
 predict_good += 1
print("测试集中的正常邮件数为", test_good, "预测对的正常邮件数为", predict_good)
print("测试集中的垃圾邮件数为", test_spam, "预测对的垃圾邮件为", predict_spam)
print("总预测正确个数为", (predict_spam + predict_good), "总正确率为",
(predict_spam + predict_good) / len(predict_Y))
