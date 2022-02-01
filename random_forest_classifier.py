# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 22:44:05 2021

@author: tilwa

This file implements the random forest classifier model using 
trees as the estimators, which are implemented in the file Estimator.py

"""

# importing the required packages including the DecisionTreeEstimator
# from Estimator.py
import pandas as pd
import random
from Estimator import DecisionTreeEstimator

class RandomForestModel:
    
    # getters and modifiers for the estimators instance
    def add_estimator(self, est):   # add a newly formed estimator in the list
        self.models.append(est)
    def get_estimators(self):       # fetch the list of estimators
        return self.models
    
    """
    # get the class prediction probabilites for the supplied test data
    # inputs = dataframe of test_data
    # returns = dataframe of predicted probabilites for each class of the training data
    """
    def predict_proba(self, test_data):
        # variable to store the predictions
        predictions = pd.DataFrame(columns = self.data_labels)
        
        # returning average probabilities from all estimators
        for est in self.get_estimators():
            # calling probability prediction of the estimator
            est_pred = est.predict_proba(test_data)
            predictions = predictions.add(est_pred, fill_value = 0)
            predictions = predictions.fillna(value = 0)
        n_estimators = len(self.get_estimators())
        # returning the average of probabilities
        return predictions.applymap(lambda x: x / n_estimators)
    
    """
    # function to predict the actual labels using the probabilities
    # inputs = test data dataframe
    # returns = dataframe of predicted labels
    """
    def predict(self, test_data):
        # getting probabilities and taking the maximum probability class as label
        predictions = self.predict_proba(test_data)
        label_df = pd.DataFrame(data = predictions.idxmax(axis = 1))
        return label_df  
    
    """
    # function to predict accuracy by predicting on test_data and comparing 
    # with supplied true labels as df
    # inputs = test data dataframe and true labels dataframe
    # returns = floating point number between 0 and 1
    """
    def predict_accuracy(self, test_data, true_labels):
        # calling predict on test data to get labels, based on class probabilties
        pred = self.predict(test_data)
        pred["truth_label"] = true_labels
        # getting accuracy after prediction
        accuracy = len(pred[pred["truth_label"] == pred[0]]) / len(pred)
        return accuracy 
    
    """
    # function to return random samples of data to be sent for training each estimator
    # inputs = total training data (train_dataset)
               min_samples - min rows(data points) in a sample
               min_features - minimum features to be included in a sample
               max_features - maximum features to be included in a sample
    # returns = dataframe of random samples 
    """
    def get_random_sample(self, train_dataset, min_samples, min_features, max_features):
        n_rows = random.randint(min_samples, len(train_dataset))
        features = list(train_dataset.columns)
        features.remove("target_col")
        random.shuffle(features)
        cols = features[0:random.randint(min_features, max_features)] + ["target_col"]
        return train_dataset.sample(n = n_rows).loc[:, cols].reset_index(drop = True)
    
    """
    # function to verify the values, supplied to the .fit function
    # errors are raised if values are not suitable
    # inputs =  training dataset (train_dataset)
                labels for the training data as dataframe (train_labels )
                column of target labels in training dataset (target_column)
                number of estimators to be trained (n_estimators)
                min_samples - min rows(data points) in a sample
                min_features - minimum features to be included in a sample
                max_features - maximum features to be included in a sample
                max_depth = maximum depth of any estimator
    # returns = Nothing
    """
    def verify_inputs(self, train_dataset, train_labels, target_column, 
            n_estimators, min_samples, min_features, max_features, max_depth):
        # assertion statements for various values on data types provided to fit function
        assert type(train_dataset) == type(pd.DataFrame())
        assert type(n_estimators) == type(int()) and n_estimators > 0
        assert type(min_samples) == type(int()) and min_samples > 0
        assert type(min_features) == type(int()) and min_features > 0
        assert max_features is None or (type(max_features) == type(int()) 
                                        and max_features > 0 and max_features > min_features)
        assert max_depth is None or (type(max_depth) == type(int()) and max_depth > 0)
        
        # exactly one of the train_labels or target_column parameters is required
        # i.e. either labels are supplied separately or as part of training data,
        # with column name given as target_column
        if (train_labels is None and target_column is None) or (train_labels is not None and target_column is not None):
            raise ValueError("Please either provide series or list of train_labels for the dataset or \
                             name of the target_column present in train_dataset")
    """    
    # fit function for the RandomForestMoel instance that takes all the parameters 
    # inputs = described above
    # returns = Nothing
    # adds the trained estimator to the class variables models
    """
    def fit(self, train_dataset, train_labels = None, target_column = None, 
            n_estimators = 5, min_samples = 5, min_features = 2, 
            max_features = None, max_depth = None):
        # verifying all the input values
        self.verify_inputs(train_dataset, train_labels, target_column, 
            n_estimators, min_samples, min_features, max_features, max_depth)
        train_dataset.reset_index(drop = True, inplace = True)
        
        # checking for exactly one value out of train_labels and target_column
        if train_labels is not None:
            assert len(train_labels) == len(train_dataset)
            # appending labels in the dataset as "target_col" column name
            train_dataset["target_col"] = train_labels
        else:
            # raise error if target_column not present in the training dataset
            if target_column not in train_dataset.columns:
                raise Exception(f"target_column {target_column} not present in train_dataset")
            # renaming the target_column as "target_col"
            train_dataset.rename(columns = {target_column: "target_col"},
                                 inplace = True)
            # setting max features as total features available, if not given
            if max_features is None:
                # excluding the labels column
                max_features = len(train_dataset.columns) - 1
        self.data_labels = list(train_dataset["target_col"].unique())
        
        # invoking __init__ to reset variables in case fit is called on a trained object
        self.__init__()
        # fitting the data on objects of decision trees and adding trained models
        # to the class instance variable models
        for est in range(0, n_estimators):
            estimator = DecisionTreeEstimator()
            # getting random sample to feed to the decision tree estimator
            data_sample = self.get_random_sample(train_dataset, min_samples, 
                                                 min_features, max_features)
            estimator.fit(data_sample, max_depth)
            self.add_estimator(estimator)
    
    def __init__(self):
        # variables to store the state of the forest model
        self.models = []   # stores the instances of different trees
        self.labels = []   # stores the total classes seen in the training data, used while prediction
        











