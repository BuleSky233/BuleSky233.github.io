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
##LinearRegression 
在线性回归中分别使用BGD和SGD的方法进行迭代，训练过程中使用BGD和SGD对应的损失函数值如图1所示，训练后对测试集的测试结果与真实值的比较如图3所示（图3对X和Y进行了标准化） 

'import random
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler'

