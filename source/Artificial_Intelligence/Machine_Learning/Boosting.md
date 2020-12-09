#### 本页目录

* [集成学习](#集成学习)
    * [Boosting 方法 - 模型融合](#Boosting%20方法%20-%20模型融合)
        * [Boosting 简介](#Boosting%20简介)
        * [Forward Stagewise Additive Modeling - 前向分步加法模型](#Forward%20Stagewise%20Additive%20Modeling%20-%20前向分步加法模型)
        * [四大 Boosting 算法](#四大%20Boosting%20算法)
        * [Adaboost (Adaptive Boosting) 算法](#Adaboost)
            * [adaboost 算法流程](#adaboost%20算法流程)
            * [adaboost 误差界分析](#adaboost%20误差界分析)
            * [adaboost 案例分析](#adaboost%20案例分析)
        * [GBDT(Gradient Boosting Decision Tree) 梯度提升](#GBDT)
            * [GBDT特征选择](#GBDT特征选择)
            * [GBDT回归树](#GBDT回归树)
            * [GBDT分类树](#GBDT分类树)
            * [GBDT 例题](#GBDT%20例题)
        * [XGBoost](#(eXtreme%20Gradient%20Boosting))
            * [GBDT特征选择](#GBDT特征选择)
            * [XGBoost 与 GBDT 区别于联系](#XGBoost%20与%20GBDT%20区别于联系)
            * [XGBoost原理](#XGBoost原理)
            * [XGBoost推导](#XGBoost推导)
            * [XGBoost 例题](#XGBoost%20例题)
            * [XGBoost 高频题 ](#XGBoost%20高频题)
    * [Bagging方法](#Bagging方法)

----
## 集成学习

* 集成学习会挑选一些简单的基础模型进行组装，组装这些基础模型的思路主要有 2 种方法：
    1. bagging（bootstrap aggregating的缩写，也称作“套袋法”）
        * 并行式集成学习方法最著名的代表
        * Bagging 的思路是所有基础模型都一致对待，每个基础模型手里都只有一票。然后使用民主投票的方式得到最终的结果。
        * 大部分情况下，经过 bagging 得到的结果方差（variance）更小。
        * 在 bagging 的方法中，最广为熟知的就是随机森林了：bagging + 决策树 = 随机森林
        * 具体过程
            1. 从原始样本集中抽取训练集。每轮从原始样本集中使用Bootstraping的方法抽取n个训练样本（在训练集中，有些样本可能被多次抽取到，而有些样本可能一次都没有被抽中）。共进行k轮抽取，得到k个训练集。（k个训练集之间是相互独立的）
            2. 每次使用一个训练集得到一个模型，k个训练集共得到k个模型。（注：这里并没有具体的分类算法或回归方法，我们可以根据具体问题采用不同的分类或回归方法，如决策树、感知器等）
            3. 对分类问题：将上步得到的k个模型采用投票的方式得到分类结果；对回归问题，计算上述模型的均值作为最后的结果。（所有模型的重要性相同）
                ![9](ML_img/9.png)
        
    2. boosting
        * Boosting 和 bagging 最本质的差别在于他对基础模型不是一致对待的，而是经过不停的考验和筛选来挑选出「精英」，然后给精英更多的投票权，表现不好的基础模型则给较少的投票权，然后综合所有人的投票得到最终结果。
        * 大部分情况下，经过 boosting 得到的结果偏差（bias）更小
        * 在 boosting 的方法中，比较主流的有 Adaboost 和 Gradient boosting 。
        * 具体过程
            1. 通过加法模型将基础模型进行线性的组合。
            2. 每一轮训练都提升那些错误率小的基础模型权重，同时减小错误率高的模型权重。
            3. 在每一轮改变训练数据的权值或概率分布，通过提高那些在前一轮被弱分类器分错样例的权值，减小前一轮分对样例的权值，来使得分类器对误分的数据有较好的效果。
                ![10](ML_img/10.png)

* Boosting 和 bagging 的异同
* 样本选择上：
    * Bagging：训练集是在原始集中有放回选取的，从原始集中选出的各轮训练集之间是独立的。
    * Boosting：每一轮的训练集不变，只是训练集中每个样例在分类器中的权重发生变化。而权值是根据上一轮的分类结果进行调整。
* 样例权重：
    * Bagging：使用均匀取样，每个样例的权重相等
    * Boosting：根据错误率不断调整样例的权值，错误率越大则权重越大。
* 预测函数：
    * Bagging：所有预测函数的权重相等。
    * Boosting：每个弱分类器都有相应的权重，对于分类误差小的分类器会有更大的权重。
* 并行计算：
    * Bagging：各个预测函数可以并行生成
    * Boosting：各个预测函数只能顺序生成，因为后一个模型参数需要前一轮模型的结果。

---
## Boosting 方法 - 模型融合

### Boosting 简介
* 对于一个复杂任务来说，将多个专家的判断进行`适当（按照一定权重）`的`综合（例如线性组合加法模型）`所得出的判断，要比其中任何一个专家单独的判断好

* 在`概率近似正确（probably approximately correct，PAC）`学习框架中：
    * 一个概念（一个类，label），如果存在一个多项式的学习算法能够学习它，并且`正确率很高`，那么就称这个概念是`强可学习的`；
    * 一个概念（一个类，label），如果存在一个多项式的学习算法能够学习它，学习的`正确率仅比随机猜测略好`，那么就称这个概念是`弱可学习的`。
    * `强可学习和弱可学习是等价的`。 也就是说，在PAC学习的框架下，一个概念是强可学习的 充分必要条件 是这个概念是弱可学习的。
    * 这样一来，问题就转化为了，在算法训练建模中，如果已经发现了“弱可学习算法”（即当前分类效果并不优秀，甚至仅仅比随机预测效果要好），就有可能将其boosting（提升）为强可学习算法，这其中最具代表性的方法就是AdaBoosting（AdaBoosting algorithm）

* Boosting 要做的事情就是把弱可学习方法变成强可学习方法，提升方法就是从弱学习算法出发，反复学习，得到一系列弱分类器（基本分类器），然后组合这些弱分类器，构成一个强分类器。大多数的提升方法都是改变训练数据的概率分布（训练数据的权重分布）

* Boosting 思路：
    对于一个学习问题来说（以分类问题为例），给定训练数据集，求一个弱学习算法要比求一个强学习算法要容易的多。Boosting方法就是从弱学习算法出发，反复学习，得到一系列弱分类器，然后组合弱分类器，得到一个强分类器。Boosting方法在学习过程中通过改变训练数据的权值分布，针对不同的数据分布调用弱学习算法得到一系列弱分类器。

* 常见的模型组合方法有：
    * 简单平均（Averaging）
    * 投票（voting）
    * Bagging（randomforest）
    * boosting（GBDT）
    * stacking
    * blending
    * 等

* Boosting设计的问题：
    1. 在每一轮学习之前，如何改变训练数据的权值分布？
        (Adaboost算法的做法是：提高那些被前一轮弱分类器错误分类样本的权值，而降低那些被正确分类样本的权值。)
    2. 如何将一组弱分类器组合成一个强分类器？
        (AdaBoost采取加权多数表决的方法。具体地：`加大分类误差率小` 的弱分类器的权值，使其在表决中起较大的作用；`减小分类误差率大`的弱分类器的权值，使其在表决中起较小的作用。)


### Forward Stagewise Additive Modeling - 前向分步加法模型

![boosting 1](ML_img/boosting_1.png)

![boosting 2](ML_img/boosting_2.png)


### 四大 Boosting 算法

![boosting 3](ML_img/boosting_3.png)

L2Boosting全称：Least Squares Boosting；该算法由Buhlmann和Yu在2003年提出。

![boosting 5](ML_img/boosting_5.png)

![boosting 4](ML_img/boosting_4.png)

----

## Adaboost (Adaptive Boosting) 算法

* 由Yoav Freund和Robert Schapire在1995年提出
* 它的自适应在于：
    前一个基本分类器分错的样本会得到加强，加权后的全体样本再次被用来训练下一个基本分类器。同时，在每一轮中加入一个新的弱分类器，直到达到某个预定的足够小的错误率或达到预先指定的最大迭代次数。

* 算法步骤
    1. 初始化训练数据的权值分布。如果有N个样本，则每一个训练样本最开始时都被赋予相同的权值：1/N。
    2. 训练弱分类器。具体训练过程中，如果某个样本点已经被准确地分类，那么在构造下一个训练集中，它的权值就被降低；相反，如果某个样本点没有被准确地分类，那么它的权值就得到提高。然后，权值更新过的样本集被用于训练下一个分类器，整个训练过程如此迭代地进行下去。
    3. 将各个训练得到的弱分类器组合成强分类器。各个弱分类器的训练过程结束后，加大分类误差率小的弱分类器的权重，使其在最终的分类函数中起着较大的决定作用，而降低分类误差率大的弱分类器的权重，使其在最终的分类函数中起着较小的决定作用。换言之，误差率低的弱分类器在最终分类器中占的权重较大，否则较小。

### adaboost 算法流程

![adaboost 6](ML_img/adaboost_6.png)

![adaboost 1](ML_img/adaboost_1.png)

![adaboost 2](ML_img/adaboost_2.png)

### adaboost 误差界分析

![adaboost 3](ML_img/adaboost_3.png)

![adaboost 4](ML_img/adaboost_4.png)

![adaboost 5](ML_img/adaboost_5.png)

### adaboost 案例分析

![adaboost 7](ML_img/adaboost_7.png)
![adaboost 8](ML_img/adaboost_8.png)
![adaboost 9](ML_img/adaboost_9.png)
![adaboost 10](ML_img/adaboost_10.png)
![adaboost 11](ML_img/adaboost_11.png)
![adaboost 12](ML_img/adaboost_12.png)


------

### GBDT(Gradient Boosting Decision Tree) 梯度提升树

* GBDT主要由三个概念组成：
    * Regression Decistion Tree（即DT)，
    * Gradient Boosting（即GB)，
    * Shrinkage (算法的一个重要演进分枝，目前大部分源码都按该版本实现）
    * 理解它如何用于搜索排序则需要额外理解RankNet概念，之后便功德圆满。

* GBDT通过多轮迭代,每轮迭代产生一个弱分类器，每个分类器在上一轮分类器的残差基础上进行训练。 弱分类器一般会选择为CART TREE（也就是分类回归树）。
* 每一轮预测和实际值有`残差`，下一轮根据残差再进行预测，最后将所有预测相加，就是结果。
* GBDT中的树都是`回归树`，`不是分类树`，这点对理解GBDT相当重要（尽管`GBDT调整后也可用于分类但不代表GBDT的树是分类树`）。

![GBDT 1](ML_img/gbdt_1.png)
![GBDT 2](ML_img/gbdt_2.png)

#### GBDT特征选择

![GBDT 3](ML_img/gbdt_3.png)

#### GBDT回归树

![GBDT 4](ML_img/gbdt_4.png)
![GBDT 5](ML_img/gbdt_5.png)
![GBDT 6](ML_img/gbdt_6.png)
![GBDT 7](ML_img/gbdt_7.png)

#### GBDT分类树

![GBDT 8](ML_img/gbdt_8.png)
![GBDT 9](ML_img/gbdt_9.png)
![GBDT 10](ML_img/gbdt_10.png)
![GBDT 11](ML_img/gbdt_11.png)
[多分类例题及代码实现](https://www.cnblogs.com/bnuvincent/p/9693190.html)

#### GBDT 例题
[例题1](https://blog.csdn.net/qq_22238533/article/details/79199605)
* 李航 统计学习例题：
![GBDT 12](ML_img/gbdt_12.png)
![GBDT 13](ML_img/gbdt_13.png)
![GBDT 14](ML_img/gbdt_14.png)

-----

### XGBoost(eXtreme Gradient Boosting)

#### XGBoost 与 GBDT 区别于联系

![XGBoost 1](ML_img/xgboost_1.png)

#### XGBoost原理

![XGBoost 2](ML_img/xgboost_2.png)
![XGBoost 3](ML_img/xgboost_3.png)
![XGBoost 4](ML_img/xgboost_4.png)
![XGBoost 5](ML_img/xgboost_5.png)
![XGBoost 6](ML_img/xgboost_6.png)
![XGBoost 7](ML_img/xgboost_7.png)
![XGBoost 8](ML_img/xgboost_8.png)
(https://blog.csdn.net/qq_22238533/article/details/79477547)

#### XGBoost推导
(https://zhuanlan.zhihu.com/p/92837676)

![XGBoost 9](ML_img/xgboost_9.png)
![XGBoost 10](ML_img/xgboost_10.png)
![XGBoost 11](ML_img/xgboost_11.png)
![XGBoost 12](ML_img/xgboost_12.png)
![XGBoost 13](ML_img/xgboost_13.png)
![XGBoost 14](ML_img/xgboost_14.png)
![XGBoost 15](ML_img/xgboost_15.png)
![XGBoost 16](ML_img/xgboost_16.png)
![XGBoost 17](ML_img/xgboost_17.png)
![XGBoost 18](ML_img/xgboost_18.png)
![XGBoost 19](ML_img/xgboost_19.png)
![XGBoost 20](ML_img/xgboost_20.png)
![XGBoost 21](ML_img/xgboost_21.png)
![XGBoost 22](ML_img/xgboost_22.png)
![XGBoost 23](ML_img/xgboost_23.png)
![XGBoost 24](ML_img/xgboost_24.png)



#### XGBoost 例题
(https://www.jianshu.com/p/ac1c12f3fba1)

#### XGBoost 高频题

![XGBoost 25](ML_img/xgboost_25.png)
![XGBoost 26](ML_img/xgboost_26.png)
![XGBoost 27](ML_img/xgboost_27.png)
![XGBoost 28](ML_img/xgboost_28.png)
![XGBoost 29](ML_img/xgboost_29.png)
![XGBoost 30](ML_img/xgboost_30.png)
![XGBoost 31](ML_img/xgboost_31.png)
![XGBoost 32](ML_img/xgboost_32.png)
![XGBoost 33](ML_img/xgboost_33.png)
![XGBoost 34](ML_img/xgboost_34.png)
![XGBoost 35](ML_img/xgboost_35.png)
![XGBoost 36](ML_img/xgboost_36.png)
![XGBoost 37](ML_img/xgboost_37.png)
![XGBoost 38](ML_img/xgboost_38.png)
![XGBoost 39](ML_img/xgboost_39.png)
![XGBoost 40](ML_img/xgboost_40.png)

------


## Bagging方法

* Bagging是并行式集成学习方法最著名的代表。
* 它基于自助采样法(bootstrap sampling)。给定包含m个样本的数据集，我们先随机取出一个样本放入采样集中，再把该样本放回到初始数据集，使得下次采样时该样本仍然由可能被采到。这样，经过m次随机采样操作，我们可以得到含有m个样本的采样集。初始训练集中有的样本被多次采到，有的则一次也没有被采样到，初始样本集中大约会有63.2%的样本出现在采样集中。
* 通过上述的自助采样法可以得到T个集合，每个集合包含m个样本。然后，基于每个集合训练出一个基学习器，再将这些基学习器进行集合。
* 在对各个基学习器的预测结果进行结合时，Bagging通常对分类任务使用简单投票法，对回归任务使用简单平均法。若分类预测时出现两个类收到同样票数的情形，则最简单的做法是随机选一个，也可以进一步考察投票的置信度来确定最终胜者。
* 从方差—偏差分解的角度看，Bagging主要关注降低方差，（高方差对应的是过拟合问题），因此它在不剪枝决策树、神经网络等易受样本扰动的学习器上效果更为明显。


![bagging 1](ML_img/bagging_1.png)


### Random Forest(RF) 随机森林

#### 随机森林简介

* 随机森林（Random Forest，RF）是bagging的一个扩展变体。RF在 以决策树为基学习器构建Bagging集成 的基础上，进一步在决策树的训练过程中引入了`随机属性选择`。具体来说，传统决策树在选择划分属性时，是在当前节点的属性集合（假定由d个集合）中选择一个最优属性；而在RF中，对基决策树的每个节点，先从该节点属性中随机选择一个包含k个属性的属性子集，然后从这个子集中选择一个最优属性用于划分。这里的参数k控制了随机性的引入程度，若k=d，则基决策树的构建与传统的决策树相同；若k=1，则是随机选择一个属性用于划分；一般情况下，推荐k=log2d。 这种随机选择属性也会使得RF的训练效率比bagging更高。

* 随机森林简单，容易实现，计算开销小，令人惊奇的是它在很多现实任务中展现出了很强大的性能，被誉为“代表集成学习技术水平方法”。 随机森林法仅仅对bagging做了小改动，但是，与bagging中基学习器的“多样性”仅仅通过样本扰动（通过对初始训练集bootstrap采样）而来不同，RF中基学习器的多样性不仅仅来自样本扰动，而且还来自属性扰动，这就使得最终集成的泛化性能可通过个体学习器之间差异度的增加而进一步提高。


#### 随机森林特点

* 在当前所有算法中，具有极好的准确率/It is unexcelled in accuracy among current algorithms；
* 能够有效地运行在大数据集上/It runs efficiently on large data bases；
* 能够处理具有高维特征的输入样本，而且不需要降维/It can handle thousands of input variables without variable deletion；
* 能够评估各个特征在分类问题上的重要性/It gives estimates of what variables are important in the classification；
* 在生成过程中，能够获取到内部生成误差的一种无偏估计/It generates an internal unbiased estimate of the generalization error as the forest building progresses；
* 对于缺省值问题也能够获得很好得结果/It has an effective method for estimating missing data and maintains accuracy when a large proportion of the data are missing
* 它就相当于机器学习领域的Leatherman（多面手），你几乎可以把任何东西扔进去，它基本上都是可供使用的。在估计推断映射方面特别好用，以致都不需要像SVM那样做很多参数的调试。

##### 随机森林算法流程(bagging + 决策树 = 随机森林)

![Random Forest 4](ML_img/rf_4.png)

###### Bagging
Bagging是并行式集成学习方法最著名的代表。它直接基于自助采样法(bootstrap sampling)
1. 从原始样本集中有放回随机抽取n个训练样本，独立进行k轮抽取，得到k个训练集
2. 独立训练k个模型（基学习器可以是：决策树，ANN等）
3. 分类问题：投票产生分类结果； 回归问题：取k个模型预测结果的均值
4. 预测函数可以并行生成


![Random Forest 1](ML_img/rf_1.png)

![Random Forest 9](ML_img/rf_9.png)

![Random Forest 3](ML_img/rf_3.png)

![Random Forest 2](ML_img/rf_2.png)


##### 随机森林算法优缺点
![Random Forest 5](ML_img/rf_5.png)

##### 随机森林算法例子

![Random Forest 6](ML_img/rf_6.png)
![Random Forest 7](ML_img/rf_7.png)
![Random Forest 8](ML_img/rf_8.png)

------










