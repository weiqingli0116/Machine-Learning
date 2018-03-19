import math
import numpy as np
from problem2 import DT
#-------------------------------------------------------------------------
'''
    Problem 3: Bagging: Boostrap Aggregation of decision trees (on continous attributes)
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.
'''


#-----------------------------------------------
class Bag(DT):
    '''
        Bagging ensemble of Decision Tree (with contineous attributes)
        Hint: Bagging is a subclass of DT class in problem2. So you can reuse and overwrite the code in problem 2.
    '''


    #--------------------------
    @staticmethod
    def bootstrap(X,Y):
        '''
            Create a boostrap sample of the dataset.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                X: the bootstrap sample of the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the bootstrap sample of the class labels, a numpy array of length n. Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        n = X.shape[1]
        newX = X.copy()
        newY = Y.copy()
        for i in range(n):
            a = np.random.random_integers(0,n-1)
            newX[:,i] = X[:,a]
            newY[i] = Y[a]
        X = newX
        Y = newY
        #########################################
        return X, Y



    #--------------------------
    def train(self, X, Y, n_tree=11):
        '''
            Given a training set, train a bagging ensemble of decision trees.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                n_tree: the number of trees in the ensemble
            Output:
                T: a list of the root of each tree, a list of length n_tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        T = []
        for tree in range(n_tree):
            X1,Y1 = Bag.bootstrap(X, Y)
            d = DT()
            T.append(d.train(X1,Y1))

        #########################################
        return T


    #--------------------------
    @staticmethod
    def inference(T,x):
        '''
            Given a bagging ensemble of decision trees and one data instance, infer the label of the instance.
            Input:
                T: a list of decision trees.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        ys = []
        for tree in T:
            ys.append(DT.inference(tree,x))
        d = {}
        for item in ys:
            if item in d.keys():
                d[item] += 1
            else:
                d[item] = 1
        y = sorted(d,key=lambda x:d[x])[-1]


        #########################################
        return y



    #--------------------------
    @staticmethod
    def predict(T,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset.
            Input:
                T: a list of decision trees.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        n = X.shape[1]
        Y = []
        for col in range(n):
            x = X[:,col]
            Y.append(Bag.inference(T,x))
        Y = np.array(Y)

        #########################################
        return Y


    #--------------------------
    @staticmethod
    def load_dataset(filename='data3.csv'):
        '''
            Load dataset 3 from the CSV file:data3.csv.
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        X = []
        Y = []
        with open(filename,'r') as f:
            f.next()
            for line in f:
                content = line.strip('\r\n').split(',')
                try:
                    Y.append(float(content[0]))
                except ValueError:
                    Y.append(content[0])
                try:
                    X.append(map(float,content[1:]))
                except ValueError:
                    X.append(content[1:])

        X = np.array(X).T
        Y = np.array(Y)



        #########################################
        return X,Y

