# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 04:21:33 2021

@author: tilwa

The following file implements the C4.5 algorithm to build a decision tree, which
can be used to make standalone decision trees or from the an instance of RandomForestModel class

"""

import pandas as pd
import math
import numpy as np

"""
# The node class is used as the datatype for each node of the tree
# Each Node stores the following:
    - distribution of labels at that node (based on dataset seen by that node during training)
    - information of branches based on boolean tests on feature values
"""
class Node:
    
    # function to calculate and store counts of classes seen in the data supplied during training
    def get_class_dist(self, data):
        return data["target_col"].value_counts()
    
    # adding branches to the current Node object based on different tests on the splitting feature
    def add_branch(self, node, partition_info):
        feat, test, val = partition_info[0:3]
        self.branches.append(((feat, test, val), node))
    
    # initialising the instance by calculating the class distribution and empty branches variable
    def __init__(self, data):
        self.class_dist = self.get_class_dist(data)
        self.branches = []   
    

"""
# Class for the decision tree, used to create decision tree object
"""
class DecisionTreeEstimator:
    
    # updating the root_node reference after the tree formation
    def update_model(self, root_node_ref):
        self.model = root_node_ref
    # getter for the root node reference
    def get_root_node(self):
        return self.model
    
    """
    # function to test condition of the test data at any node position, to direct it to further branches
    # input = row of the test data dataframe
              test - test stored as the property of that Node to go deep in the tree
    # returns = Boolean value true or false based on passing/failing the test condition
    """
    def test_condition(self, row, test):
        feat, condition, val = test
        # checking for different test types (which can be stored as condition of the Node)
        try:
            if condition == "equals":
                return row[feat] == val
            if condition == "less_than_equal":
                return row[feat] <= val
            if condition == "greater_than":
                return row[feat] > val
        except KeyError:
            return False
        return False
    
    """
    # function to return class probabilities at a leaf node
    # inputs = node instance (object of class Node)
    # returns = dictionary of probabilities of different classes
    """
    def dist_prob(self, node):
        # accessing the class_dist information of the node
        total_examples = sum(node.class_dist)
        label_prob = {}
        # calculating probabilities of each class in the distribution
        for label, counts in node.class_dist.iteritems():
            label_prob[label] = counts / total_examples
        return label_prob
    
    """
    # function to result class probabilites for a particular row of data (test case) based on current node position
    # inputs = row (test case)
               node (position at which the prediction currently is)
    # returns =  probabilities of classes as dictionaries, after recursively testing conditions and sending 
                the example to different nodes until any leaf is reached
    """
    def result(self, row, node):
        # accessing the branch information of the node
        branches = node.branches
        # condition for leaf node (branches are zero)
        if len(branches) == 0:
            return self.dist_prob(node)     # return probability of classes according to current node
        # test for conditions to direct to other nodes of branches
        for branch in branches:
            # call result recursively for next node if the test condition is passed
            if self.test_condition(row, branch[0]):
                return self.result(row, branch[1])
        # in case of a missing feature value, all branches will be exhausted
        # and class probabilities according to current node will be returned
        return self.dist_prob(node)
    
    """
    # function to predict class probabilities for each data point of the supplied test data
    # input = test data as dataframe
    # returns = dataframe of predicted probabilites for each class 
    """
    def predict_proba(self, test_data):
        predictions = pd.DataFrame(columns = self.data_labels)  # variable to store predictions
        # getting root node reference
        root_node = self.get_root_node()
        for index, row in test_data.iterrows():
            # getting result for each example
            predictions = predictions.append(self.result(row, root_node), ignore_index = True)
        return predictions.fillna(value = 0)
    
    """
    # function to calculate entropy of the data
    # inputs = labels of the dataset
    # returns = entropy as a floating point number 
    """
    def get_entropy(self, target_col):
        Ent = 0
        n_data_points = len(target_col)
        # using value counts ofr each class
        for count in target_col.value_counts():
            # calculating class probability in the target labels
            class_prob = count / n_data_points
            # updating entropy value
            Ent += -1 * class_prob * math.log2(class_prob)
        return Ent
    
    """
    # function to get the ratio for a particular split (for the data group based on condition on feature vals)
    # inputs = iterable data_groups
               ent_data - entropy of the overall dataset from which groups are derived
               n - total number of datapoints of the overall data
    # returns = gain ratio for the group of data points
    """
    def conditioned_entropy(self, data_groups, ent_data, n):
        ent_post_partition, info_used = 0, 0
        for group in data_groups:
            if len(group) == 0:
                continue
            feat_prob = len(group) / n
            # calculating post partition entropy
            ent_post_partition += self.get_entropy(group["target_col"]) * feat_prob
            # calculating the information used
            info_used += -1 * feat_prob * math.log2(feat_prob)
        # information gain
        gain = ent_data - ent_post_partition
        if info_used == 0: return 0
        return gain / info_used
    
    """
    # function to calculate gain ratio for a feature given the dataset
    # inputs = data, feature (name of the feature)
    # returns = return the tuple for best gain ratio for a feature, 
    i.e. tuple of val and gain ratio in case of continuous attribute
    and tuple of single gain ratio value for continuous attribute
    """
    def calc_information_gain_ratio(self, data, feature):
        # calculating entropy of given dataset
        ent_data = self.get_entropy(data["target_col"])
        n_data_points = len(data)
        # for continuous variable
        if data[feature].dtype in [np.dtype("float64"), np.dtype("int64")]:
            feat_vals = np.array(data[feature])
            # sorting the values
            feat_vals.sort()
            # getting all split candidates for continuous variable
            split_cand = (feat_vals[0:-1] + feat_vals[1:]) / 2
            best_gain_ratio, gain_ratio = -1, ()
            # calculating gain ratio for each split value and keeping the best value
            for val in split_cand:
                val_gain_ratio = self.conditioned_entropy((data[data[feature] <= val], 
                                      data[data[feature] > val]), ent_data, n_data_points)
                if val_gain_ratio > best_gain_ratio:
                    best_gain_ratio = val_gain_ratio
                    gain_ratio = (val, val_gain_ratio)
            
        else:
            data_groups = (group[1] for group in data.groupby(by = feature))
            gain_ratio = (self.conditioned_entropy(data_groups, ent_data, n_data_points), )
        return gain_ratio
    
    
    """
    # function to calculate gain ratios and return the best split feature name
    """
    def get_split_feature(self, data):
        features = list(data.columns)
        # if no featrues left (only target column left)
        if len(features) == 1: return None
        # removing target column from list of features
        features.remove("target_col")
        # calculating gain ratios for all features 
        # and returning the feature information for the maximum gain ratio
        gain_ratios = [(feat, self.calc_information_gain_ratio(data, feat)) for feat in features]
        return max(gain_ratios, key = lambda x: x[1][-1])
    
    """
    # function to return partitions of data based on values to split the data on
    # inputs = data
               feature_info = name and value of feature to split upon 
               (value to split on in case of continuous attributes)
               categorical = boolean value denoting if the feature is categorical
    # returns = list of data partitions combined with the property and tests,
                which will be used to define further branches
    """
    def get_data_partitions(self, data, feature_info, categorical):
        # getting feature name to partition on
        feat = feature_info[0]
        # splitting on number of unique values in case of categorical variable
        if categorical: 
            # tests defined as equals
            partitions = list(map(lambda x: (feat, "equals", x[0], x[1]), list(data.groupby(by = feat))))
        else:
            # partitioning on best split value for a continuous attribute
            feat_val = feature_info[1][0]
            # tests defined as less_than_equal or greater_than
            partitions = [(feat, "less_than_equal", feat_val, data[data[feat] <= feat_val]), 
                      (feat, "greater_than", feat_val, data[data[feat] > feat_val])]
        return list(map(lambda x: (x[0], x[1], x[2], x[3].drop(columns = [feat])), partitions))
    
    """
    # function to build tree recursively based on best attributes to split
    # inputs = train data (data)
               node = node reference where subtree will be attached via branch
               depth = current depth of the tree
               max_depth = maximum allowed depth of the tree
    # returns = node with the further subtree attached
    """
    def build_tree(self, node, data, depth, max_depth):
        # checking for base conditions (if depth == max_depth, or data is pure)
        if len(data) == 0 or data["target_col"].nunique() == 1 or (max_depth is not None and depth == max_depth):
            return node
        # getting best attribute to split the data
        best_att_info = self.get_split_feature(data)
        # returning current node if the features list is exhausted
        if best_att_info is None:
            return node
        # identifying if the feature is categorical or continuous, to calculate gain ratios
        if len(best_att_info[1]) == 2: 
            categorical = False 
        else: categorical = True
        # getting data partitions for further branching (2 partitions for continuous variables
                                # and multiple for categorical variables)
        data_subsets = self.get_data_partitions(data, best_att_info, categorical)
        # building subtree recursively for each partition
        for partition in data_subsets:
            subtree = self.build_tree(Node(partition[3]), partition[3], depth + 1, max_depth)
            # adding the subtree returned to the current node
            node.add_branch(subtree, partition)
        return node
    
    # function to fit data by building tree and updating the root node reference
    def fit(self, data, max_depth):
        # storing the labels seen in the training data
        self.data_labels = list(data["target_col"].unique())
        # calling the build tree method
        root_node = self.build_tree(Node(data), data, 0, max_depth)
        # storing the object of root node
        self.update_model(root_node)
    
    def __init__(self):
        # variables to store the state of the forest model
        self.model = None     # variable to store reference to root node of the formed tree
        self.data_labels = [] # data_labels seen by the tree in the training dataset, 
                        # used to return probabilities of all class
    
    
    
    
    
    
    
    
    
    
    