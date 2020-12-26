
### Deep Learning
* [深度学习 - 神经网络](#)
    * [Basic knowledge](#Basic%20knowledge)
    * [Specific artificial neural networks](#Specific artificial neural networks)
        * [Linear Threshold Unit (or Perceptron) - 感知机 又称 基于LTN的人工神经元](#)
        * [Multilayer Perceptrons - 多层感知机](#Multilayer%20Perceptrons%20-%20多层感知机)
        * [Back Propagation Neural Network - 反向传播神经网络](#Back%20Propagation%20Neural%20Network%20-%20反向传播神经网络)
            * [激活函数](#激活函数)
            * [bp算法](#bp算法)
            * [反向传播神经网络计算过程](#反向传播神经网络计算过程)
            * [反向传播总结](#反向传播总结)
        * [Convolutional Neural Networks - 卷积神经网络](#Convolutional%20Neural%20Networks%20-%20卷积神经网络)
            * [基本结构](#基本结构)
            * [CNN训练方法](#CNN训练方法)
            * [CNN的卷积层](#CNN的卷积层)
            * [CNN的池化层 - 数据降维,避免过拟合](#CNN的池化层%20-%20数据降维,避免过拟合)
            * [CNN全连接网络](#CNN全连接网络)
            * [CNN 公式计算](#CNN%20公式计算)
            * [CNN 例子](#CNN%20例子)
            * [CNN 框架发展](#CNN%20框架发展)
        * [AlexNet](#AlexNet)
        * [Restricted Boltzmann Machines - 受限玻尔兹曼机](#Restricted%20Boltzmann%20Machines%20-%20受限玻尔兹曼机)
            * [RBM 计算](#RBM%20计算)
            * [RBM 数学原理](#RBM%20数学原理)
            * [RBM 例子](#RBM%20例子)
        * [Deep Belief Network 深度置信网络](#Deep%20Belief%20Network%20深度置信网络)
        * [Autoencoder Networks - 自编码网络](#Autoencoder%20Networks%20-%20自编码网络)
            * [稀疏自编码器推导](#稀疏自编码器推导)
        * [generic artificial neural network - 遗传神经网络](#generic%20artificial%20neural%20network%20-%20遗传神经网络)
        * [Recurrent Neural Network 循环神经网络](#Recurrent%20Neural%20Network%20循环神经网络)
            * [循环神经网络推导求解](#循环神经网络推导求解)
            * [推导理解](#推导理解)
            * [基于时间的反向传播理解](#基于时间的反向传播理解)
            * [tanh梯度消失的问题](#tanh梯度消失的问题)
        * [Long short-term memory(LSTM) 长短时记忆网络](#)
            * [推导](#推导)
        * [Recursive Neural Network 递归神经网络](#Recursive%20Neural%20Network%20递归神经网络)
        * [Graph Neural Networks(GNN) 图神经网络](#)
        * [Competitive Learning Networks - 竞争学习神经网络](#Competitive%20Learning%20Networks%20-%20竞争学习神经网络)
        * [待补充](#)
            * [Attention机制](#Attention机制)
            * [Inhibitory Feedback Networks - 抑制反馈网络](#Inhibitory%20Feedback%20Networks%20-%20抑制反馈网络)
            * [Radial Basis Function Networks - 径向基函数神经网络](#Radial%20Basis%20Function%20Networks%20-%20径向基函数神经网络)
            * [Hopfield Networks - 一种单层反馈神经网络](#Hopfield%20Networks%20-%20一种单层反馈神经网络)
            * [Kohonen Networks - 自组织特征映射 SOM(Self-organizing feature Map）](#)
            * [Capsule Networks - 胶囊网络](#Capsule%20Networks%20-%20胶囊网络)

-----

### Basic knowledge

##### ` Layers` can be:
* `visible` – receive inputs from, or send outputs to, the external environment
* `hidden` – only receive inputs and send output to other processing units
* `input layers`: receive signals from the environment – for classification, this is the feature vector which is to be classified
* `output layers`: which send signals to the environment  – for classification, this is the predicted class label associated with the feature vector

##### `Weights`   -    Connection weights can be defined by
* Setting weights explicitly using prior knowledge.
* Optimising connectivity to achieve some objective (e.g. using a genetic algorithm).
* Training the network by feeding it training data and allowing it to adapt the connection weights.
    * supervised    -   Delta Learning Rule
    * unsupervised  -   Hebbian learning rule

----

### Specific artificial neural networks
----
* #### Linear Threshold Unit (or Perceptron) - 感知机 又称 基于LTN的人工神经元
    * Each unit receives a vector of inputs, x (from other units or the external environment).
    * Each input is associated with a weight, w, which determines the strength of influence of each input.
    * Each w can be either +ve (excitatory) or -ve (inhibitory).
    * (restricted case, where weights and activations are binary, known as Threshold Logic Unit, or McCulloch-Pitts neuron)
    ![Perceptron rule](img/Perceptron_rule.png)

    * This `response function` often split into two component parts:
        * A `transfer function` that determines how the inputs are integrated.
        * An `activation function` that determines the output the neuron produces.
    
    * `w` and `θ` definea hyperplane that divides the input space into two parts
    * This `hyperplane` is called the “`decision boundary`.”

    ![Perceptron rule](img/decision_boundary.png)

----

* #### Multilayer Perceptrons - 多层感知机
![Multilayer Perceptrons](img/mlp.png)
----
* #### Back Propagation Neural Network - 反向传播神经网络
    * ##### 激活函数
    ![Back Propagation Neural Network 31](img/bp_31.png)
    * ##### bp算法
     ![Back Propagation Neural Network 28](img/bp_28.png)
     ![Back Propagation Neural Network 29](img/bp_29.png)
     ![Back Propagation Neural Network 30](img/bp_30.png)

    * ##### 反向传播神经网络计算过程
    ![Back Propagation Neural Network 32](img/bp_32.png)
    ![Back Propagation Neural Network 1](img/bp_1.png)
    ![Back Propagation Neural Network 2](img/bp_2.png)
    ![Back Propagation Neural Network 3](img/bp_3.png)
    ![Back Propagation Neural Network 4](img/bp_4.png)
    ![Back Propagation Neural Network 5](img/bp_5.png)
    ![Back Propagation Neural Network 6](img/bp_6.png)
    ![Back Propagation Neural Network 7](img/bp_7.png)
    ![Back Propagation Neural Network 8](img/bp_8.png)
    ![Back Propagation Neural Network 9](img/bp_9.png)
    ![Back Propagation Neural Network 10](img/bp_10.png)
    ![Back Propagation Neural Network 11](img/bp_11.png)
    ![Back Propagation Neural Network 12](img/bp_12.png)
    ![Back Propagation Neural Network 13](img/bp_13.png)
    ![Back Propagation Neural Network 14](img/bp_14.png)
    ![Back Propagation Neural Network 15](img/bp_15.png)
    ![Back Propagation Neural Network 16](img/bp_16.png)
    ![Back Propagation Neural Network 17](img/bp_17.png)
    ![Back Propagation Neural Network 18](img/bp_18.png)
    ![Back Propagation Neural Network 19](img/bp_19.png)
    ![Back Propagation Neural Network 33](img/bp_33.png)
    ![Back Propagation Neural Network 20](img/bp_20.png)
    
    * ##### 反向传播总结
    ![Back Propagation Neural Network 21](img/bp_21.png)
    ![Back Propagation Neural Network 22](img/bp_22.png)
    ![Back Propagation Neural Network 23](img/bp_23.png)
    ![Back Propagation Neural Network 24](img/bp_24.png)
    ![Back Propagation Neural Network 25](img/bp_25.png)
    ![Back Propagation Neural Network 26](img/bp_26.png)
    ![Back Propagation Neural Network 27](img/bp_27.png)

----

* #### Convolutional Neural Networks - 卷积神经网络
https://cs231n.github.io/convolutional-networks/
* 基本结构
![CNN 1](img/cnn_1.png)
![CNN 2](img/cnn_2.png)
* CNN训练方法
![CNN 4](img/cnn_4.png)
    

* CNN的卷积层
![CNN 3](img/cnn_3.png)
![CNN 11](img/cnn_11.png)
![CNN 14](img/cnn_14.png)
![CNN 15](img/cnn_15.png)
![CNN 16](img/cnn_16.png)
![CNN 17](img/cnn_17.png)
![CNN 18](img/cnn_18.png)
![CNN 19](img/cnn_19.png)
![CNN 22](img/cnn_22.png)
![CNN 23](img/cnn_23.png)
![CNN 24](img/cnn_24.png)

* CNN的池化层 - 数据降维,避免过拟合
池化层相比卷积层可以更有效的降低数据维度，这么做不但可以大大减少运算量，还可以有效的避免过拟合。

![CNN 10](img/cnn_10.png)
![CNN 20](img/cnn_20.png)


* CNN全连接网络
![CNN 12](img/cnn_12.png)
![CNN 13](img/cnn_13.png)


* CNN 公式计算
训练方式为反向传播
![CNN 25](img/cnn_25.png)
![CNN 26](img/cnn_26.png)
![CNN 27](img/cnn_27.png)
![CNN 28](img/cnn_28.png)
![CNN 29](img/cnn_29.png)
![CNN 30](img/cnn_30.png)
![CNN 31](img/cnn_31.png)
![CNN 32](img/cnn_32.png)

* CNN 例子
![CNN 5](img/cnn_5.png)
![CNN 6](img/cnn_6.png)
![CNN 7](img/cnn_7.png)
![CNN 8](img/cnn_8.png)
![CNN 9](img/cnn_9.png)


* CNN 框架发展

![CNN 21](img/cnn_21.png)

----

* #### AlexNet

![AlexNet 12](img/alexnet_12.png)
![AlexNet 1](img/alexnet_1.png)
![AlexNet 2](img/alexnet_2.png)
![AlexNet 3](img/alexnet_3.png)
![AlexNet 4](img/alexnet_4.png)
![AlexNet 5](img/alexnet_5.png)
![AlexNet 6](img/alexnet_6.png)
![AlexNet 7](img/alexnet_7.png)
![AlexNet 8](img/alexnet_8.png)
![AlexNet 9](img/alexnet_9.png)
![AlexNet 10](img/alexnet_10.png)
![AlexNet 11](img/alexnet_11.png)

----

* #### Restricted Boltzmann Machines - 受限玻尔兹曼机
https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf
https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf

![RBM 1](img/rbm_1.png)
![RBM 2](img/rbm_2.png)
![RBM 3](img/rbm_3.png)
![RBM 4](img/rbm_4.png)
![RBM 5](img/rbm_5.png)
![RBM 6](img/rbm_6.png)
![RBM 7](img/rbm_7.png)
![RBM 8](img/rbm_8.png)
![RBM 9](img/rbm_9.png)
![RBM 10](img/rbm_10.png)
![RBM 11](img/rbm_11.png)
![RBM 12](img/rbm_12.png)

* RBM 计算
![RBM 13](img/rbm_13.png)
![RBM 14](img/rbm_14.png)
![RBM 15](img/rbm_15.png)
![RBM 16](img/rbm_16.png)
![RBM 17](img/rbm_17.png)

* RBM 数学原理
![RBM 23](img/rbm_23.png)
![RBM 24](img/rbm_24.png)
![RBM 25](img/rbm_25.png)
![RBM 26](img/rbm_26.png)
![RBM 27](img/rbm_27.png)

* RBM 例子
![RBM 18](img/rbm_18.png)
![RBM 19](img/rbm_19.png)
![RBM 20](img/rbm_20.png)
![RBM 21](img/rbm_21.png)
![RBM 22](img/rbm_22.png)

![RBM 28](img/rbm_28.png)
![RBM 29](img/rbm_29.png)
![RBM 30](img/rbm_30.png)
![RBM 31](img/rbm_31.png)
![RBM 32](img/rbm_32.png)

-----
* #### Deep Belief Network 深度置信网络
https://s-top.github.io/blog/2018-01-19-dbn
![DBN 3](img/dbn_3.png)
![DBN 1](img/dbn_1.png)
![DBN 2](img/dbn_2.png)

![DBN 4](img/dbn_4.png)
![DBN 5](img/dbn_5.png)
![DBN 6](img/dbn_6.png)
![DBN 7](img/dbn_7.png)
![DBN 8](img/dbn_8.png)
![DBN 9](img/dbn_9.png)

-----

* #### Autoencoder Networks - 自编码网络
https://zhuanlan.zhihu.com/p/54032437
![AE 1](img/ae_1.png)
![AE 2](img/ae_2.png)
![AE 3](img/ae_3.png)
![AE 4](img/ae_4.png)
![AE 14](img/ae_14.png)

* 稀疏自编码器推导

![AE 5](img/ae_5.png)
![AE 6](img/ae_6.png)
![AE 7](img/ae_7.png)
![AE 8](img/ae_8.png)
![AE 9](img/ae_9.png)
![AE 10](img/ae_10.png)
![AE 11](img/ae_11.png)
![AE 12](img/ae_12.png)
![AE 13](img/ae_13.png)

-----

* #### generic artificial neural network - 遗传神经网络

* An `environment` in which the system operates (inputs to the network, outputs from the network).
* A set of `processing units` (‘neurons’, ‘cells’, ‘nodes’).
* A set of `weighted connections` between units, wji, which determines the influence of unit i on unit j.
* A `transfer function` that determines how the inputs to a unit are integrated.
* An `activation function` that determines the output the neuron produces.
* An `activation state`, yj, for every unit (‘response’, ‘output’).
* A `method` for setting/changing the connection weights.
------

* #### Recurrent Neural Network 循环神经网络
https://zhuanlan.zhihu.com/p/32755043
![RNN 6](img/rnn_6.png)
![RNN 1](img/rnn_1.png)
![RNN 2](img/rnn_2.png)
![RNN 3](img/rnn_3.png)
![RNN 4](img/rnn_4.png)
![RNN 5](img/rnn_5.png)
![RNN 18](img/rnn_18.png)
![RNN 19](img/rnn_19.png)

![RNN 7](img/rnn_7.png)
![RNN 8](img/rnn_8.png)
![RNN 9](img/rnn_9.png)
![RNN 10](img/rnn_10.png)
![RNN 11](img/rnn_11.png)
![RNN 12](img/rnn_12.png)
![RNN 13](img/rnn_13.png)
![RNN 14](img/rnn_14.png)
![RNN 15](img/rnn_15.png)
![RNN 16](img/rnn_16.png)
![RNN 17](img/rnn_17.png)

* 循环神经网络推导求解
https://www.cnblogs.com/YiXiaoZhou/p/6058890.html

![RNN 20](img/rnn_20.png)
![RNN 21](img/rnn_21.png)
![RNN 22](img/rnn_22.png)
![RNN 23](img/rnn_23.png)

* 推导理解
![RNN 24](img/rnn_24.png)
![RNN 25](img/rnn_25.png)
![RNN 26](img/rnn_26.png)
![RNN 27](img/rnn_27.png)
![RNN 28](img/rnn_28.png)
![RNN 29](img/rnn_29.png)
![RNN 30](img/rnn_30.png)
![RNN 31](img/rnn_31.png)
![RNN 32](img/rnn_32.png)

* 基于时间的反向传播理解
![RNN 33](img/rnn_33.png)
![RNN 34](img/rnn_34.png)
![RNN 35](img/rnn_35.png)
![RNN 36](img/rnn_36.png)

* tanh梯度消失的问题

![RNN 37](img/rnn_37.png)
![RNN 38](img/rnn_38.png)

----

* #### Long short-term memory(LSTM) 长短时记忆网络
循环神经网络(RNN) 的优化算法
![LSTM 16](img/lstm_16.png)
![LSTM 7](img/lstm_7.png)
![LSTM 8](img/lstm_8.png)
![LSTM 9](img/lstm_9.png)
![LSTM 10](img/lstm_10.png)
![LSTM 11](img/lstm_11.png)
![LSTM 12](img/lstm_12.png)
![LSTM 13](img/lstm_13.png)
![LSTM 14](img/lstm_14.png)
![LSTM 15](img/lstm_15.png)

![LSTM 6](img/lstm_6.png)
![LSTM 1](img/lstm_1.png)
![LSTM 2](img/lstm_2.png)
![LSTM 3](img/lstm_3.png)
![LSTM 4](img/lstm_4.png)
![LSTM 17](img/lstm_17.png)
![LSTM 18](img/lstm_18.png)
![LSTM 19](img/lstm_19.png)

* 推导

![LSTM 20](img/lstm_20.png)
![LSTM 21](img/lstm_21.png)
![LSTM 22](img/lstm_22.png)
![LSTM 23](img/lstm_23.png)
![LSTM 24](img/lstm_24.png)
![LSTM 25](img/lstm_25.png)
![LSTM 26](img/lstm_26.png)
![LSTM 27](img/lstm_27.png)
![LSTM 28](img/lstm_28.png)
![LSTM 29](img/lstm_29.png)
![LSTM 30](img/lstm_30.png)
![LSTM 31](img/lstm_31.png)
![LSTM 32(img/lstm_32.png)
![LSTM 33(img/lstm_33.png)
![LSTM 34](img/lstm_34.png)
![LSTM 35](img/lstm_35.png)
![LSTM 36](img/lstm_36.png)
![LSTM 37](img/lstm_37.png)
![LSTM 38](img/lstm_38.png)
![LSTM 39](img/lstm_39.png)
![LSTM 40](img/lstm_40.png)
![LSTM 41](img/lstm_41.png)
![LSTM 42](img/lstm_42.png)
![LSTM 43](img/lstm_43.png)
![LSTM 44](img/lstm_44.png)
![LSTM 45](img/lstm_45.png)
![LSTM 46](img/lstm_46.png)
![LSTM 47](img/lstm_47.png)
![LSTM 48](img/lstm_48.png)
![LSTM 49](img/lstm_49.png)



------

* #### Recursive Neural Network 递归神经网络

![RNN 39](img/rnn_39.png)
![RNN 40](img/rnn_40.png)
![RNN 41](img/rnn_41.png)
![RNN 42](img/rnn_42.png)
![RNN 43](img/rnn_43.png)
![RNN 44](img/rnn_44.png)
![RNN 45](img/rnn_45.png)
![RNN 46](img/rnn_46.png)

----------


* #### Graph Neural Networks(GNN) 图神经网络

清华大学GNN paper list: https://github.com/thunlp/GNNPapers/blob/master/README.md#survey-papers

![GNN 1](img/GNN_1.png)
![GNN 2](img/GNN_2.png)
![GNN 3](img/GNN_3.png)
![GNN 4](img/GNN_4.png)
![GNN 5](img/GNN_5.png)
![GNN 6](img/GNN_6.png)
![GNN 7](img/GNN_7.png)
![GNN 8](img/GNN_8.png)
![GNN 9](img/GNN_9.png)
![GNN 10](img/GNN_10.png)
![GNN 11](img/GNN_11.png)
![GNN 12](img/GNN_12.png)
![GNN 13](img/GNN_13.png)
![GNN 14](img/GNN_14.png)
![GNN 15](img/GNN_15.png)
![GNN 16](img/GNN_16.png)
![GNN 17](img/GNN_17.png)
![GNN 18](img/GNN_18.png)
![GNN 19](img/GNN_19.png)
![GNN 20](img/GNN_20.png)
![GNN 21](img/GNN_21.png)


----
* #### Competitive Learning Networks - 竞争学习神经网络

所谓竞争学习神经网络，是在输出层对多个节点返回的数据做选择。选择方法可以是投票可以是算平均值等。

-----

* #### Attention机制

-----

* #### Inhibitory Feedback Networks - 抑制反馈网络

* #### Radial Basis Function Networks - 径向基函数神经网络

* #### Hopfield Networks - 一种单层反馈神经网络

* #### Kohonen Networks - 自组织特征映射 SOM(Self-organizing feature Map）

* #### Capsule Networks - 胶囊网络

-----







