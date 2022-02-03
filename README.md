# random-forests-from-scratch
This is an implementation of random forests from complete scratch in Python. Includes many customized features to use.

## Implementation Details

For implementation, following three classes are used to store the states of any random forest model.

2.1.RandomForestModel: This is the class to be used while training the model. Each instance of this class stores estimators along with the total labels seen in the training data. This class is contained in the code file random_forest_classifier.py. Some of the important methods are as follows:
Method	Description
fit	fit(self, train_dataset, train_labels = None, target_column = None, n_estimators = 5, min_samples = 5, min_features = 2, max_features = None, max_depth = None)

Inputs:
train_dataset - training dataset
train_labels - labels for the training data as dataframe
target_column - column of target labels in training dataset
n_estimators - number of estimators to be trained
min_samples - min allowable rows(data points) in a sample
min_features - minimum allowable features to be included in a sample
max_features - maximum allowable features to be included in a sample
max_depth = maximum allowable depth of any estimator

Function: Fits the supplied data to the model.
get_random_sample	get_random_sample(self, train_dataset, min_samples, min_features, max_features)

Function: returns a random sample of data, to send to decision tree.
predict_proba	predict_proba(self, test_data)

Function: returns the probabilities of classes for all instances in test dataframe. Also referred when predict and predict_accuracy are called.
Table 1: Main Methods in RandomForestModel class

2.2.DecisionTreeEstimator: This class is part of the file Estimator.py and is used to create an instance of the decision tree (using C4.5 algorithm). It is called from RandomForestModel but can also be used standalone to create decision trees. Some important methods of this class are:
Method	Description
fit	fit(self, data, max_depth)

Inputs:
data - training dataset
max_depth - maximum allowable depth

Function: Fits the supplied data to the model.
build_tree	build_tree(self, node, data, depth, max_depth)

Inputs:
data - training dataset available at the level (provided to node)
node - node reference where subtree will be attached via branch
depth - depth of the tree
max_depth - maximum allowable depth

Function: Recursively creates tree and returns the node reference 
get_split_feature	get_split_feature(self, data)

Function: function to calculate gain ratios and return best feature.
calc_information_gain_ratio	calc_information_gain_ratio(self, data, feature)

Function: returns information gain ratio for the dataset splitted over the given feature. Invoked by get_split_feature.
predict_proba	predict_proba(self, test_data)

Function: returns probabilities of classes for all instances in test_data 
Table 2: Main methods in DecisionTreeEstimator class

2.3.Node: Part of Estimator.py, this class is used to store a node of the tree with the class probabilities and rules required to go ahead to any of the branches. The class implements methods add_branch to connect two nodesand get_class_dist that returns the distribution of classes at the current node.