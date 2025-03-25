'''

First we initialize the utility functions

'''
from itertools import chain, combinations
from statistics import mean
import numpy as np
import entropy_estimators as ee
import pandas as pd
from sklearn.metrics import accuracy_score

def argmax(lst):
    """ Returns the position of maximal element of the list """
    return lst.index(max(lst))

def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)



def powerset(iterable):
    '''
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    '''
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))[1:]

class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, columns, column, value):
        self.columns = columns
        self.column = column
        self.value = value

    def match(self, example):
        ''' Compare the feature value in an example to the
         feature value in this question. '''
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            self.columns[self.column], condition, str(self.value))

def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return np.array(true_rows), np.array(false_rows)

def gini(rows):
    """Calculate the Gini Impurity for a list of rows.

    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

def find_best_split_inf_gain(rows, columns):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature
        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(columns, col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_question

##### Banzhaf Part starts




def isWinning(player, coalition, threshold=0.05):
    ''' Checks if inclusion of the player in the coalition
        leads to winning situation.
        The player wins if the player is interdependent on
        atleast half of the members in the coalition.
        The interdependence is measured using conditional mutual information.
        Inputs: player: Pandas dataframe
         coalition: Pandas dataframe
         threshold: boundary value of interdependence
        Outputs: Boolean which returns True if inclusion leads to winning coalition.
    '''
    total_dependence = 0
    x = player.values.reshape(-1, 1).tolist()
    if coalition.shape[1] == 1:
        return ee.mi(x, coalition.values.reshape(-1, 1).tolist()) >= threshold

    for i in range(0, coalition.shape[1]):
        y = coalition.drop(coalition.columns[i], axis=1).values.tolist()
        z = coalition[coalition.columns[i]].values.reshape(-1, 1).tolist()
        if ee.cmi(x, y, z) >= threshold:
            total_dependence = total_dependence + 1
    return float(total_dependence)/float(len(coalition)) >= threshold

def banzhaf(df, column, threshold=0.05):
    '''
    Calculates the banzhaf power index of the feature whose data is given by the argument column
    Inputs: df: Pandas dataframe
            column_name: Pandas dataframe
    Outputs: Banzhaf Power Index
    '''
    assert column.columns[0] not in df.columns, "Player should not be part of coalition"
    assert column.shape[1] == 1, "Only one player is allowed"
    all_coalitions = powerset(df.columns)
    positive_swing = 0
    for coalition in all_coalitions:
        coalition = list(coalition)
        if isWinning(column, df[coalition], threshold):
            positive_swing = positive_swing + 1
    banzhaf_index = float(positive_swing) / (len(all_coalitions))
    return round(banzhaf_index, 4)

def find_best_split_banzhaf(data):
    """Find the best question to ask by calculating the banzhaf power index
    and selecting the maximum value from it """
    banzhaf_rows = []
    for i in range(data.shape[1]):
        df = data.drop(data.columns[i], axis=1)
        column = data[data.columns[i]].to_frame()
        banzhaf_rows.append(banzhaf(df, column))
    idx_max = argmax(banzhaf_rows)
    mean_val = mean(unique_vals(data.values, idx_max))
    question = Question(data.columns, idx_max, mean_val)
    return question


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)

def build_tree(data, isbanzhaf=False, depth=3):
    """ Builds the decision tree
    Inputs: data: the training dataset
            banzhaf: True if user wants Banzhaf Decision Tree
            depth: depth of decision tree desiried
    Outputs: Decision node
    """
    rows = data.values
    if depth <= 0:
        return Leaf(rows)
    question = find_best_split_inf_gain(rows, data.columns)
    true_rows, false_rows = partition(rows, question)
    if true_rows.size == 0:
        true_branch = Leaf(rows)
    if false_rows.size == 0:
        false_branch = Leaf(rows)
    if true_rows.size and false_rows.size:
        true = pd.DataFrame(true_rows, columns=data.columns)
        false = pd.DataFrame(false_rows, columns=data.columns)
        if isbanzhaf:
            true_branch = build_banzhaf_tree(true, depth=depth-1)
            false_branch = build_banzhaf_tree(false, depth=depth-1)
        else:
            true_branch = build_tree(true, depth=depth-1)
            false_branch = build_tree(false, depth=depth-1)
    return Decision_Node(question, true_branch, false_branch)


def build_banzhaf_tree(rows, depth=1):
    ''' Builds pure Banzhaf Decision Tree
    '''
    if depth <= 0:
        return Leaf(rows.values)
    question = find_best_split_banzhaf(rows)
    true_rows, false_rows = partition(rows.values, question)
    if true_rows.size == 0:
        true_branch = Leaf(rows)
    if false_rows.size == 0:
        false_branch = Leaf(rows)
    if true_rows.size and false_rows.size:
        true = pd.DataFrame(true_rows, columns=rows.columns)
        false = pd.DataFrame(false_rows, columns=rows.columns)
        true_branch = build_banzhaf_tree(true, depth=depth-1)
        false_branch = build_banzhaf_tree(false, depth=depth-1)
    return Decision_Node(question, true_branch, false_branch)

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return max(node.predictions, key=node.predictions.get)

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    return classify(row, node.false_branch)

def fetch_probability(row, node):
    """ Fetches the Probability"""
     # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return fetch_probability(row, node.true_branch)
    return fetch_probability(row, node.false_branch)
    
def select_kbest_features(k, data):
    """
    Implements the algorithm given in Section 3.5
    """
    k = k +1
    output = list(data.iloc[:, -1].values.reshape(-1, 1))
    S = []
    while k > 0:
        curr_info = 0
        MI = []
        columns = list(data.columns.values)
        for f in columns[:-1]:
            if len(S) == 0:
                y = list(data[f].values.reshape(-1, 1))
                curr_info = ee.mi(output,y)
            else:
                if f in S:
                    continue
                else:
                    y = list(data[f].values.reshape(-1,1))
                    z = list(data[S].values.reshape(-1,len(S)))
                    curr_info = ee.cmi(output,y,z)
            if curr_info == 0:
                data.drop(f, axis=1, inplace=True)
            else:
                MI.append({'value':curr_info,'label':f})
        maxMIfeature = max(MI, key=lambda x:x['value'])
        S.append(maxMIfeature['label'])
        k = k - 1
    return S


class DecisionTree:
    '''
    This class is the main decision tree
    '''
    def __init__(self, isBanzhaf):
        self.is_banzhaf = isBanzhaf
        self.data = None
        self.tree = None
        self.output_vals = None
    def fit(self, X, y, depth=2):
        """ Fits the decision tree
        """
        self.data = pd.concat([X, y], axis=1)
        self.tree = build_tree(self.data, self.is_banzhaf, depth)
        self.output_vals = unique_vals(y.values, 0)
    def print_the_tree(self):
        """
        Prints the tree
        """
        print_tree(self.tree)
    def predict(self, x_test):
        """
        Predicts the class
        """
        return classify(x_test, self.tree)
    def predict_proba(self, x_test):
        """
        Predicts the class probabilities in the order of output vals
        """
        prob_dict = {}
        for out_vals in self.output_vals:
            prob_dict[out_vals] = 0.0
        class_dict = fetch_probability(x_test, self.tree)
        sum_value = 0.0
        for key, value in class_dict.items():
            sum_value = sum_value + value
            prob_dict[key] = value
        for key, value in prob_dict.items():
            prob_dict[key] = round(prob_dict[key] / sum_value , 4)
        return prob_dict



    def score(self, y_pred, X_test):
        """
        Returns the score
        """
        predictions = []
        for x_test in X_test:
            predictions.append(self.predict(x_test))
        return accuracy_score(y_pred, predictions)