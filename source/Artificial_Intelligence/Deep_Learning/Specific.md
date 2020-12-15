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
* #### Linear Threshold Unit (or Perceptron)   - 感知机 又称 基于LTN的人工神经元
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

* #### Multilayer Perceptrons    -   多层感知机
![Multilayer Perceptrons](img/mlp.png)
----
* #### Back Propagation Neural Network  -   反向传播神经网络
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

* #### generic artificial neural network    -   遗传神经网络

* An `environment` in which the system operates (inputs to the network, outputs from the network).
* A set of `processing units` (‘neurons’, ‘cells’, ‘nodes’).
* A set of `weighted connections` between units, wji, which determines the influence of unit i on unit j.
* A `transfer function` that determines how the inputs to a unit are integrated.
* An `activation function` that determines the output the neuron produces.
* An `activation state`, yj, for every unit (‘response’, ‘output’).
* A `method` for setting/changing the connection weights.
------

* #### Competitive Learning Networks -   竞争学习神经网络



* #### Inhibitory Feedback Networks  -   抑制反馈网络


* #### Autoencoder Networks  -   自编码网络


* #### Radial Basis Function Networks    -   径向基函数神经网络


* #### Convolutional Neural Networks -   卷积神经网络


* #### Restricted Boltzmann Machines -   受限玻尔兹曼机


* #### Hopfield Networks -   一种单层反馈神经网络


* #### Kohonen Networks  -  自组织特征映射 SOM(Self-organizing feature Map）


* #### Capsule Networks  -   胶囊网络




-----







