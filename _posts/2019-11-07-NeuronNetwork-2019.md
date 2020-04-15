---
layout:     post                    # 使用的布局（不需要改）
title:      Project3:Neuron Network               # 标题 
subtitle:   assignment of Machine Learning course #副标题
date:       2019-11-07             # 时间
author:     Zongyou Liu                      # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - project 
    - machine learning
---

## 实验背景：
图像识别在现实生活中具有非常大的实践意义，可以应用于自动驾驶、机器人等。手写数字的识别作为深度学习的成功案例之一，被应用于手写邮政编码识别以实现自助寄件。可见，图像识别能带给我们的生活许多便利，所以，基于图像的神经网络算法的学习至关重要。
## 实验方法：
使用深度学习框架tensorflow进行神经网络的搭建
## 实验过程：
### Ⅰ.单层神经网络
1. 数据预处理  
下载数据，并划分训练集和测试集
2. 搭建神经网络框架，使用单层神经网络，即只有输入层和以softmax作为激活函数的输出层  
3. 用训练集对神经网络进行训练以学习参数  
4. 用测试集对模型进行测试以检验模型的正确性 
![neuron1](https://raw.githubusercontent.com/BuleSky233/BuleSky233.github.io/master/img/neuron1.png)  
正确率约为0.91，证明分类效果一般，所以我试着通过增加神经网络的层数来提高分类效果
### Ⅱ.三层神经网络
1. 数据预处理  
下载数据，并划分训练集和测试集     
2. 搭建神经网络框架，使用三层神经网络，即有第0层输入层、第1层Flatten层负责将二维图像展成一维，不带学习参数，第2层带学习参数的隐藏层、第3层输出层（以softmax作为激活函数）    
3. 用训练集对神经网络进行训练以学习参数  
   训练过程总结如下：    
   迭代次数|准确率
   --|--:
   第一次|0.9153
   第二次|0.9573
   第三次|0.9678
   第四次|0.9729
   第五次|0.9763
 
   程序运行的部分截图：
   ![neuron2](https://raw.githubusercontent.com/BuleSky233/BuleSky233.github.io/master/img/neuron2.png)
   ![neuron3](https://raw.githubusercontent.com/BuleSky233/BuleSky233.github.io/master/img/neuron3.png)
   ![neuron4](https://raw.githubusercontent.com/BuleSky233/BuleSky233.github.io/master/img/neuron4.png)
4. 用测试集对模型进行测试以检验模型的正确性
   ![neuron5](https://raw.githubusercontent.com/BuleSky233/BuleSky233.github.io/master/img/neuron5.png)
   可以看出，通过增加神经网络的层数，我们达到的准确率约为0.98，证明分类效果不错。
   下面我选择测试集中的第一个样本来具体看看测试情况：
   该样本的图像为：
   ![neuron6](https://raw.githubusercontent.com/BuleSky233/BuleSky233.github.io/master/img/neuron6.png)
   输出层输出10个数字，第i个数字代表0-9中数字i的给分，其中最大的即为神经网络的预测结果，如下图所示：
   ![neuron7](https://raw.githubusercontent.com/BuleSky233/BuleSky233.github.io/master/img/neuron7.png)
   可见，神经网络预测的结果与实际情况相符，这一样本预测正确
## 实验结果：
使用单层神经网络进行分类对测试集的正确率达到0.91，使用三层神经网络进行分类对测试集的正确率达到0.98。 
## 结果讨论：
从实验过程可看出，神经网络的分类性能是十分不错的，单层神经网络就可达到0.91的正确率。并且，正确率会随着神经网络层数的增加而提高，神经网络层数越多，神经元数量越多，则能学习到更加复杂的函数，可以实现更加强大的分类功能。但是层数和神经元数量的增加也意味着计算量的大大增加，一个复杂的神经网络可能需要几天的时间才能训练完成，使用GPU可以大大加速运算。
## 源代码：
单层神经网络:   
```
import tensorflow as tf
import tensorflow.compat.v1 as tfc
import matplotlib.pyplot as plt
import tensorflow_core.examples.tutorials.mnist.input_data as input_data

def main():
    tf.compat.v1.disable_eager_execution()
    # data = tf.keras.datasets.mnist
    # (x_train, y_train), (x_test, y_test) = data.load_data()

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # plt.imshow(x_train[1], cmap="binary")
    # plt.show()
    sess = tfc.InteractiveSession()
    x = tfc.placeholder("float", shape=[None, 784])
    y_ = tfc.placeholder("float", shape=[None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    sess.run(tfc.initialize_all_variables())
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    cross_entropy = -tf.reduce_sum(y_ * tfc.log(y))
    train_step = tfc.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("测试集的正确率为",accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    main()
```
三层神经网络:  
```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
print("在测试集中")
model.evaluate(x_test,  y_test, verbose=2)
plt.imshow(x_test[0], cmap="binary")
plt.show()
prediction = model.predict(x_test)
print("输出层的输出\n",prediction[0])
print("模型的预测结果为(即输出的10个数中最大数的索引为）",np.argmax(prediction[0]))
print("该测试样本的真实标签为",y_test[0])
```






