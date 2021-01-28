## 机器学习数学基础
------
### 目录
* [概率统计](#概率统计)
    * [概率核心理论](#概率核心理论)
    * [核心的几种随机变量的分布以及变量之间的关系](#核心的几种随机变量的分布以及变量之间的关系)
    * [参数估计理论](#参数估计理论)
    * [随机理论的相关概念](#随机理论的相关概念)
    * [信息论](#信息论)
    * [随机过程初步理论和应用](#随机过程初步理论和应用)
    * [时间序列分析](#时间序列分析)
* [线性代数](#线性代数)
    * [矩阵基本计算](#矩阵基本计算)
    * [行列式](#行列式)
    * [矩阵的逆矩阵](#矩阵的逆矩阵)
    * [矩阵求导](#矩阵求导)


-------

### 概率统计

#### 概率核心理论
[主要参考 及 例子](https://www.shangyang.me/2019/03/21/math-probability-04-conditional-probability/)


* ##### 经典分布
![Distribution 1](img/distribution_1.png)

[分布参考](https://blog.csdn.net/qq_28168421/article/details/103998235)

均匀分布（连续）伯努利分布（离散）二项分布（离散）多伯努利分布，分类分布（离散）多项式分布（离散）
β分布（连续）Dirichlet 分布（连续）伽马分布（连续）指数分布（连续）高斯分布（连续）正态分布（连续）
卡方分布（连续）t 分布（连续）



* ##### 期望 方差
![Probability 24](img/Probability_24.png)

* ##### 条件概率 和 全概率
![Probability 1](img/Probability_1.png)
![Probability 2](img/Probability_2.png)
![Probability 3](img/Probability_3.png)
![Probability 4](img/Probability_4.png)
![Probability 5](img/Probability_5.png)
![Probability 6](img/Probability_6.png)
![Probability 7](img/Probability_7.png)
![Probability 8](img/Probability_8.png)
![Probability 9](img/Probability_9.png)
![Probability 10](img/Probability_10.png)
![Probability 11](img/Probability_11.png)
![Probability 12](img/Probability_12.png)

* ##### 联合概率 和 边缘概率
![Probability 13](img/Probability_13.png)
![Probability 14](img/Probability_14.png)
![Probability 15](img/Probability_15.png)

* ##### 贝叶斯公式
![Probability 16](img/Probability_16.png)
![Probability 17](img/Probability_17.png)
![Probability 18](img/Probability_18.png)
![Probability 19](img/Probability_19.png)
![Probability 20](img/Probability_20.png)
![Probability 21](img/Probability_21.png)
[肝癌检测报告 例子](https://www.shangyang.me/2019/03/21/math-probability-04-conditional-probability/)
![Probability 22](img/Probability_22.png)

* ##### 朴素贝叶斯
![Probability 23](img/Probability_23.png)
[朴素贝叶斯参考](https://www.cnblogs.com/pinard/p/6069267.html)
[朴素贝叶斯参考](https://www.pkudodo.com/2018/11/21/1-3/)
[朴素贝叶斯参考](https://www.bioinfo-scrounger.com/archives/737/)
[朴素贝叶斯参考](https://blog.csdn.net/fisherming/article/details/79509025)




#### 核心的几种随机变量的分布以及变量之间的关系
分布的期望、方差等数字特征，了解概率密度函数和累积分布函数。对多组不同的变量，熟悉协方差以及相关性的意义和计算方法。


#### 参数估计理论
需要重点掌握最小偏差无偏估计、最大似然估计和贝叶斯估计的相关内容。并且学习EM算法。

#### 随机理论的相关概念
掌握蒙特卡罗方法的基本思想。同时巩固贝叶斯的思想方法，接触一下马尔科夫蒙特卡洛（MCMC）算法，找一找处理实际问题的感觉。

#### 信息论
学习关于熵的一些理论，联合熵、条件熵、交叉熵、相对熵、互信息等概念，以及最大熵模型。
[参考](https://www.cnblogs.com/qizhou/p/12178082.html)

![Entropy 1](img/Entropy_1.png)
![Entropy 7](img/Entropy_7.png)
![Entropy 2](img/Entropy_2.png)
![Entropy 3](img/Entropy_3.png)
![Entropy 4](img/Entropy_4.png)
![Entropy 5](img/Entropy_5.png)
![Entropy 6](img/Entropy_6.png)
![Entropy 8](img/Entropy_8.png)
![Entropy 9](img/Entropy_9.png)
![Entropy 10](img/Entropy_10.png)
![Entropy 11](img/Entropy_11.png)
![Entropy 12](img/Entropy_12.png)
![Entropy 13](img/Entropy_13.png)
![Entropy 14](img/Entropy_14.png)
![Entropy 15](img/Entropy_15.png)
![Entropy 16](img/Entropy_16.png)







#### 随机过程初步理论和应用
首先马尔科夫链是必须学习的，了解状态转移矩阵、多步转移、几种不同的状态分类、平稳分布等最基本的内容。然后在此概念基础上，学习隐马尔科夫链的相关内容，聚焦其基本概念，以及概率计算和参数学习的一些方法。

#### 时间序列分析
重点是移动平均、相关性以及预测等内容。




-----------


### 线性代数

#### 矩阵基本计算
![Matrix 1](img/matrix_1.png)
![Matrix 2](img/matrix_2.png)
![Matrix 3](img/matrix_3.png)
![Matrix 4](img/matrix_4.png)
![Matrix 5](img/matrix_5.png)
![Matrix 6](img/matrix_6.png)
![Matrix 7](img/matrix_7.png)

#### 行列式
![Matrix 8](img/matrix_8.png)
![Matrix 9](img/matrix_9.png)

#### 矩阵的逆矩阵
![Matrix 10](img/matrix_10.png)
![Matrix 11](img/matrix_11.png)
![Matrix 12](img/matrix_12.png)
![Matrix 13](img/matrix_13.png)
![Matrix 14](img/matrix_14.png)
![Matrix 15](img/matrix_15.png)
![Matrix 16](img/matrix_16.png)
![Matrix 17](img/matrix_17.png)
![Matrix 18](img/matrix_18.png)
![Matrix 19](img/matrix_19.png)
![Matrix 20](img/matrix_20.png)
![Matrix 21](img/matrix_21.png)
![Matrix 22](img/matrix_22.png)
![Matrix 23](img/matrix_23.png)
![Matrix 24](img/matrix_24.png)
![Matrix 25](img/matrix_25.png)
![Matrix 26](img/matrix_26.png)
![Matrix 27](img/matrix_27.png)
![Matrix 28](img/matrix_28.png)
![Matrix 29](img/matrix_29.png)
![Matrix 30](img/matrix_30.png)

#### 矩阵求导
![Matrix 31](img/matrix_31.png)
![Matrix 32](img/matrix_32.png)
![Matrix 33](img/matrix_33.png)
![Matrix 34](img/matrix_34.png)
![Matrix 35](img/matrix_35.png)
![Matrix 36](img/matrix_36.png)
![Matrix 37](img/matrix_37.png)
![Matrix 38](img/matrix_38.png)

--------






