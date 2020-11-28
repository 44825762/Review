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



* #### generic artificial neural network    -   遗传神经网络

* An `environment` in which the system operates (inputs to the network, outputs from the network).
* A set of `processing units` (‘neurons’, ‘cells’, ‘nodes’).
* A set of `weighted connections` between units, wji, which determines the influence of unit i on unit j.
* A `transfer function` that determines how the inputs to a unit are integrated.
* An `activation function` that determines the output the neuron produces.
* An `activation state`, yj, for every unit (‘response’, ‘output’).
* A `method` for setting/changing the connection weights.


* #### Competitive Learning Networks -   竞争学习神经网络



* #### Inhibitory Feedback Networks  -   抑制反馈网络


* #### Autoencoder Networks  -   自编码网络


* #### Multilayer Perceptrons    -   多层感知机



* #### Radial Basis Function Networks    -   径向基函数神经网络


* #### Convolutional Neural Networks -   卷积神经网络


* #### Restricted Boltzmann Machines -   受限玻尔兹曼机


* #### Hopfield Networks -   一种单层反馈神经网络


* #### Kohonen Networks  -  自组织特征映射 SOM(Self-organizing feature Map）


* #### Capsule Networks  -   胶囊网络


* #### Back Propagation Neural Network  -   反向传播神经网络


-----

#### 训练方式   (权重更新方式)

* Delta Learning Rule （supervised）
* Hebbian learning rule （unsupervised）





