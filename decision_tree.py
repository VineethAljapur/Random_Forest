from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}
        self.depth = 1
        
    def max_element(self, values):
        return sorted(values)[int(len(values)/2)]

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        max_ig = 0
        max_index = 0
        
        if y.count(y[0]) == len(y):
            self.tree['is_leaf'] = True
            self.tree['label'] = y[0]
            return

        if self.depth > 100:
            self.tree['is_leaf'] = True
            self.tree['label'] = self.max_element(y)
            return

        for index in range(len(X[0])):
            X_left, X_right, y_left, y_right = partition_classes(X, y, index, np.mean(X, axis=0)[index])
            ig = information_gain(y, [y_left, y_right])
            if ig > max_ig:
                max_ig = ig
                max_index = index
        X_left, X_right, y_left, y_right = partition_classes(X, y, max_index, np.mean(X, axis=0)[max_index])
        
        tig  = information_gain(y, [y_left, y_right])

        if tig < 0.001:
            self.tree['is_leaf'] = True
            self.tree['label'] = self.max_element(y)
            return
            
        if len(X_left) == len(X) or len(X_right) == len(X):
            self.tree['is_leaf'] = True
            self.tree['label'] = self.max_element(y)
            return
        else:
            self.tree['is_leaf'] = False
            self.tree['split_attr'] = max_index
            self.tree['split_val'] = np.mean(X, axis=0)[max_index]
            self.depth += 1
            self.tree['left'] = self.learn(X_left, y_left)
            self.tree['right'] = self.learn(X_right, y_right)
            return self.tree['label']
            
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)


    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        tree = self.tree
        while tree['is_leaf'] != True:
            feature = tree['split_attr']
            if record[feature] <= tree['split_val']:
                tree = self.tree['left']
            else:
                tree = self.tree['right']
        return tree['label']
