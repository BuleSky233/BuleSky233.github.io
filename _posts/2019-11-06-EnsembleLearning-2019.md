--- 
layout:     post                    # 使用的布局（不需要改）
title:      Project4:Ensemble Learning               # 标题 
subtitle:   assignment of Machine Learning course #副标题
date:       2019-11-06             # 时间
author:     Zongyou Liu                      # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - project 
    - machine learning
    - data mining
--- 


## 算法流程图：
![ensemble1](https://raw.githubusercontent.com/BuleSky233/BuleSky233.github.io/master/img/ensemble1.png)
## 算法过程：
该集成式分类器集成了三个分类算法，分别是kNN、SVM和Decision Tree。采用投票的方式将三个分类器的结果综合起来，少数服从多数，如果其中至少两个分类算法都判定样本为A类，则最终判定样本为A类。若三个分类器判断的结果均不同，则随机选取某一分类算法的结果作为最终结果。
## 实验过程：
1. 数据预处理  
   对数据进行了标准化，并划分训练集和测试集
2. 用训练集训练集成式分类器
3. 用测试集对集成式分类器进行检验

## 实验结果：
![ensemble2](https://raw.githubusercontent.com/BuleSky233/BuleSky233.github.io/master/img/ensemble2.png)
## 对比实验：
单个分类器的分类准确率如下图所示：
![ensemble3](https://raw.githubusercontent.com/BuleSky233/BuleSky233.github.io/master/img/ensemble3.png)
单个分类器的分类具体评价指标(精确率、召回率等)如下图所示：
![ensemble4](https://raw.githubusercontent.com/BuleSky233/BuleSky233.github.io/master/img/ensemble4.png)
![ensemble5](https://raw.githubusercontent.com/BuleSky233/BuleSky233.github.io/master/img/ensemble5.png)
![ensemble6](https://raw.githubusercontent.com/BuleSky233/BuleSky233.github.io/master/img/ensemble6.png)
可以看出，由kNN、SVM、decision tree集成的分类器的分类效果要好于它们各自单独的分类效果。可以看出，SVM的分类效果要明显强于kNN和decision tree，而集成式分类器的效果又相比SVM有微小的提高。
## 结果分析：
该集成式分类器具有不错的分类性能，对于本实验数据表准确率达到了0.95，且相比各个分类器单独的分类效果都有提高，但所选用的三个分类器分类效果本身就有不小的差距，所以分类准确率的提高相比于分类性能最好的SVM来说并不显著。如果选用的均为分类效果一般的分类器则可能可以更加发挥集成式分类器的优点，获得更加显著的分类效果提高。
## 收获与感悟：
本次实验使我对kNN、SVM、decision tree等算法的原理更加了解，并学会了如何构建一个集成式分类器，对数据分类的整个流程更加熟悉。
## 源代码：
CollectiveClassifier.py:
```
from sklearn import neighbors, svm
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import prettytable as pt

class CollectiveClassifier:
    def __init__(self):
        self.knnmodel = neighbors.KNeighborsClassifier()
        self.svmmodel = svm.SVC()
        self.dtmodel = DecisionTreeClassifier()

    def fit(self, train_X, train_Y):
        self.knnmodel.fit(train_X, train_Y)
        self.svmmodel.fit(train_X, train_Y)
        self.dtmodel.fit(train_X, train_Y)

    def predict(self, test_X):
        knnpredict = self.knnmodel.predict(test_X)
        svmpredict = self.svmmodel.predict(test_X)
        dtpredict = self.dtmodel.predict(test_X)
        collectivepredict = []
        for i in range(len(knnpredict)):
            if knnpredict[i] == dtpredict[i]:
                collectivepredict.append(knnpredict[i])
            else:
                collectivepredict.append(svmpredict[i])
        return collectivepredict

    def score(self, test_X, test_Y):

        knnpredict = self.knnmodel.predict(test_X)
        svmpredict = self.svmmodel.predict(test_X)
        dtpredict = self.dtmodel.predict(test_X)
        print("knn的准确率为",self.knnmodel.score(test_X, test_Y))
        print("svm的准确率为",self.svmmodel.score(test_X, test_Y))
        print("decision tree的准确率为",self.dtmodel.score(test_X, test_Y))

        print("knn各类别的评价指标为")
        print(classification_report(test_Y, knnpredict))
        print("svm各类别的评价指标为")
        print(classification_report(test_Y, svmpredict))
        print("decision tree各类别的评价指标为")
        print(classification_report(test_Y, dtpredict))

        collectivepredict = self.predict(test_X)
        dict = {}
        dict["BC_CML"] = 0
        dict["CP_CML"] = 1
        dict["k562"] = 2
        dict["normal"] = 3
        dict["pre_BC"] = 4
        label_name = ["BC_CML", "CP_CML", "k562", "normal", "pre_BC"]
        label_num = np.zeros(5)

        # 混淆矩阵 5*5
        confusion = np.zeros([5, 5])
        evaluation = np.zeros([5, 4])
        for i in range(len(collectivepredict)):
            confusion[dict[test_Y[i]]][dict[collectivepredict[i]]] += 1
            label_num[dict[test_Y[i]]] += 1
        sum = 0
        for i in confusion[0]:
            sum += i

        result = []
        right = 0

        for i in range(5):
            cur = []
            cur.append(label_name[i])
            colsum = 0
            rowsum = 0
            for j in range(5):
                colsum += confusion[j][i]
            for j in range(5):
                rowsum += confusion[i][j]
            right += confusion[i][i]
            precision = confusion[i][i] / colsum
            recall = confusion[i][i] / rowsum
            f1_score = (2 * precision * recall) / (precision + recall)
            support = label_num[i]
            cur.append(precision)
            cur.append(recall)
            cur.append(f1_score)
            cur.append(support)
            result.append(cur)

        acc=right/len(test_Y)
        print("集成式分类器的准确率为",acc)

        tb = pt.PrettyTable()
        tb.field_names = ["type", "precision", "recall", "f1-score", "support"]
        for i in range(5):
            tb.add_row(result[i])
        print("集成式分类器各类别的评价指标为")
        print(tb)
``` 
CollectiveClassification.py:
``` 
# -*- coding: utf-8 -*-
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import CollectiveClassifier

f = open('单细胞测序数据.csv')
df = pd.read_csv(f)
feature = df.values.tolist()
f.close()
f = open('单细胞测序数据Labels.csv')
df = pd.read_csv(f, header=None)
label = df.values.tolist()
ytem = []
for i in label:
    ytem.append(i[0])
label = ytem
# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(feature, label, test_size=0.3, stratify=label)
# 标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
collectivemodel = CollectiveClassifier.CollectiveClassifier()
collectivemodel.fit(X_train,Y_train)
collectivemodel.score(X_test,Y_test)
```


