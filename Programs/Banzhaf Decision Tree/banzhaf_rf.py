"""This module defines the functions required for the implementation of banzhaf random forests."""
import operator
from collections import Counter
from banzhaf_dt import DecisionTree, select_kbest_features
from sklearn.utils import resample
import pandas as pd
import numpy as np

def generate_preference(pred_proba, labels):
    '''
    Accepts:
        pred_proba: Numpy array containing probabilities
        labels: list containing output labels
    Returns: Preference in form of list of list
    '''
    num_class = pred_proba.shape[0]
    vote_dic = {}
    for i in range(num_class):
        vote_dic[labels[i]] = pred_proba[i]
    sorted_x = sorted(vote_dic.items(), key=operator.itemgetter(1))
    sorted_x.reverse()
    preference = []
    for i in range(num_class):
        preference.append(sorted_x[i][0])
    return list(preference)

def borda(preference_ballot):
    '''
    Accepts: list of list => preference_ballot
    Returns: Winner
    '''
    counts = {}
    candidates = list(set(preference_ballot[0]))
    max_point = len(candidates)
    for i in range(max_point):
        counts[candidates[i]] = 0
    for pref in preference_ballot:
        for i in range(len(pref)):
            counts[pref[i]] += (max_point -i)
    return max(counts, key=counts.get)

def get_prediction_borda(pred_proba_list, labels):
    '''
    Gets the Borda Winner
    Inputs: pred_proba: A numpy array
            labels: a list containing the output values
    '''
    preference_ballot = []
    for prob_list in pred_proba_list:
        preference_ballot.append(generate_preference(prob_list, labels))
    return borda(preference_ballot)

class BanzhafRandomForest:
    '''
    A Banzhaf Random Forest Library
    '''
    def __init__(self,num_of_trees,tree_depth):
        self.ensemble = [DecisionTree(isBanzhaf=True) for _ in range(num_of_trees)]
        self.depth = tree_depth
        self.considered_features = []
        self.labels = None
    def fit(self, X, y, depth=2):
        """
        Generates Bootstrap sampling, does feature selection and makes the ensemble
        of Banzhaf Decision Trees
        """
        self.labels = list(y[y.columns[0]].unique())
        dataframe = pd.concat([X,y],axis=1)
        for i in range(len(self.ensemble)):
            dataframe_bl = resample(dataframe, 
                                    replace=True,     # sample with replacement
                                    n_samples=int((dataframe.shape[0])*0.66))
            if dataframe.shape[1] <=5:
                selected_features = list(X.columns)
                self.considered_features.append(selected_features)
            else:
                selected_features = select_kbest_features(min(dataframe.shape[1],5), dataframe_bl) #select five features or lower
                self.considered_features.append(selected_features)
            selected_features.append(y.columns[0])
            dataframe_bl = dataframe_bl[selected_features]
            X_bl = dataframe_bl[dataframe_bl.columns[:-1]]
            y_bl = dataframe_bl[dataframe_bl.columns[-1]].to_frame()
            self.ensemble[i].fit(X_bl, y_bl, depth=depth)
            selected_features.pop(-1)
            print(i+1, 'Trees Built!')
    
    def predict(self, x_test, isBorda=True):
        """
        Generates the prediction of Banzhaf Random Forests
        Inputs: x_test: The testing sample
                isBorda: True when Borda Count should be used for Prediction
        """
        if isBorda:
            pred_prob_list = []
            for i in range(len(self.ensemble)):
                x_test_bl = x_test[self.considered_features[i]]
                pred_prob_list.append(list(self.ensemble[i].predict_proba(x_test_bl.values).values()))
            pred_prob_list = np.array(pred_prob_list)
            return get_prediction_borda(pred_prob_list, self.labels)
        predictions = []
        for i in range(len(self.ensemble)):
            x_test_bl = x_test[self.considered_features[i]]
            predictions.append(self.ensemble[i].predict(x_test_bl.values))
        occ = Counter(predictions)
        return int(max(occ, key=occ.get))




