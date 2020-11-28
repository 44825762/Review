## Data mining: 

* The process of discovering patterns in data.
* The process must be automatic or semi-automatic.
* The patterns must be meaningful and useful, i.e. lead to some benefit and inform future decisions.
    * A pattern is a series of data that repeat in a recognisable way.
* Data mining works on existing data, i.e. data that has already been generated, by people, machines, processes, etc.

### Concepts

* `Nominal`
* `Score`: quantifies how well the model performs on a data set.
* `Training set` for building a set of models.
* `Test set` for evaluating the model on unseen data.
* Positive examples are labelled with the right answer or capture the relationships of interest.
* Negative examples: are labelled with the wrong answer or contain relationships other than the ones we are looking for.
* `Feature engineering` is the process of transforming raw data into features that better represent the underlying problem and result in improved model accuracy on unseen data.
* Decision Tree - Node: corresponds to a decision to be made.
* Decision Tree - Branches from a node represent possible choices.
* Clustering - non-overlapping, i.e. each instance is in exactly one cluster,
* Clustering - overlapping, i.e. an instance may appear in multiple clusters.


### Patterns

* Allow making non-trivial predictions for new data.
* Can be expressed as: 
    * `Black boxes`, i.e. incomprehensible with hidden structure.
    * `Transparent boxes` with visible structure.
* Structural patterns:
    * Capture and explain data aspects in an explicit way.
    * Can be used for informative decisions.
    * E.g. rules in the form if-then-else.

### Real-World data Examples 

* PageRank assigns measures to web pages, based on online search query relevance (Google).
* Email filtering classifies new messages as spams or hams.
* Online advertising based on users with similar purchases.
* Social media identify users with similar preferences.

### Process of data mining

* Data mining is a process for exploring large amounts of data to discover meaningful patterns and rules.
1. Receives a data set as input.
2. Investigates for a specific type of model.
3. Produces a knowledge representation as output.

* `Step 1`: Determine the objective of the analysis: Identify the data mining problem type.
    * Supervised learning:
        * There is a target attribute.
        * If nominal, then classification. E.g. to play or not in the weather data set.
        * If numerical, then predition. E.g. predict power value in the CPU performance data set. 
    * Unsupervised learning:
        * There is no target attribute.
        * Cluster data into groups of similar objects.
        * Find correlations or associations.
    * There are other data mining tasks for other types of data.

* `Step 2`: Understand the data.
    * Visualise the data, e.g. using histograms or scatter plots.
    * Confirm that the objective can be achieved with the data set.
    * begin with the data and then select an appropriate method.

* `Step 3`: Clean and prepare the data.
    * Fix any problems with the data.
    * Consider `noise` or `missing values`
    * COnsider the data is `representative` or `biased`
    * Consider whether the data is enough
        * If the data is scarce, then data mining may not be effective.
        * A rule of thumb is that the more, the better.
        * However, very large data sets can be problematic when (i) the target variable appears in extremely rare patterns, or (ii) model building is very resource consuming.
    
* `Step 4`: Build the models.
    * Select the most appropriate model for the data.

* `Step 5`: Evaluate the model.
    * Assess whether the model achieves its goals.
    * mesure the accuracy of model
    * How well does the model described the observed data? 
    * Is the model comprehensible?

* `Step 6`: Iterate
    * Usually, multiple iterations of the above steps are required to build a good enough model.
    * Revise the performed steps, adapt and repeat.


### Data Set Attributes

* Numeric: Continuous or discrete with well-defined distance between values.
* Nominal: Categorical.
* Dichotomous: Binary or boolean or yes/no.
* Ordinal: Ordered but without well-defined distance, e.g. poor, reasonable, good and excellent health quality.
* Interval: Ordered, but also measured in fixed units, e.g. cool, mild and hot temperatures.
* Ratio: Measurements with a zero point, e.g. distance of objects from one object. The statement that a distance is three times larger than another distance makes sense.

### Data Set Issues

* may require the following steps:
    * Assembly
    * Integration
    * Cleaning
    * Transformation
* The following issues may need to be resolved: 
    * Sparsity.
    * Missing values.
    * Noise


### Data Mining Tasks

1. `Classification` models the relationship between data elements to predict classes or labels.
    * The data is classified, e.g. UK voters can be labelled as remain or leave.
    * Models ways that attributes determine the `class` of instances.
    * `Supervised learning` task because it is based on already classified instances.
    
2. `Numeric prediction` models the relationship between data elements to predict numeric quantities.
    * Models ways that attributes determine a `numeric value`.
    * Variant of classification, but without discrete classes.
    * The produced model is often more interesting than predicted values, e.g. how prices of cars are affected by their features.
    * `Supervised learning` task because the predicted outcome is known for the data set that we use to build the model.

3. `Clustering` models the relationship of data items based on their similarity to define groups and assign instances to them.
    * Models ways that instances are similar or different from each other and ways that they can be `grouped`.
    * Two instances within the same cluster should be similar.
    * Two instances in different clusters should be different.
    * Bylabellingtheclusters,wemayusetheminmeaningfulways.
    * E.g. segment customers into groups.
    * `Unsupervised learning` task because the data set does not have labels.

4. `Association` models relationships between attributes with relevant rules.
    * Models how `some attributes` determine `other attributes`.
    * No specific class or label.
    * May examine any subset of attributes to predict any other disjoint subset attributes.
    * Usually involve only nominal data.
    * E.g. use supermarket data, to identify combinations of products that occur together in transactions.
    
### Knowledge Representations

* Tables

* Trees 
    * Decision Tree
        * Typically, each branch decision: is made for a single attribute of the data set.
        * Missing Value Problem:
            * Possible solutions, all these solutions propagate errors, especially when the number of missing values increases.:
                * Ignore all instances with missing values.
                * Each attribute may get the value missing.
                * Set the most popular choice for each missing attribute value.
                * Make a probabilistic (weighted) choice for each missing attribute value, based on the other instances.
    * Functional Tree
        * Computesafunctionofmultipleattributevaluesineachnode. 
        * Branches based on the value returned by the function.
    * Regression Tree
        * Predicts numeric values.
        * Each node branches on the value of an attribute or on the value of a function of the attributes.
        * A leaf specifies a predicted value for corresponding instances.
    * Model Trees
    * Similar to a regression tree, except that a `regression equation` predicts the numeric output value in each leaf.
    * A regression equation predicts a numeric quantity as a function of the attributes.
    * More sophisticated than linear regression and regression trees.
    
* Rules
    * An expression in if-then format.
    * Conjunction (and), Disjunction (or), General logic expressions (and/or).
* Linear models
    * A linear model is a weighted sum of attribute values.
    * E.g.PRP=2.47·CACH+37.06.
    * All attribute values must be numeric.
    * Typically visualised as a 2D scatter plot with a regression line, 
    * Linear models can be applied to `classification problems`, by defining `decision boundaries` separating instances that belong to different classes.

* Instance-based representations 
* Clusters
    * Dendrogram (Hierarchical Clustering)
* Networks


### Model Evaluation

* How good is the model?
* How does it perform on known data?
* How well does it predict for new data?
* A scoring function or error function computes the differences between the predictions and the actual outcome.
* Different data mining tasks use different score functions.
* Typically, we want to maximize score or minimize error.



















