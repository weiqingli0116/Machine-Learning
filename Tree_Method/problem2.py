import math
import numpy as np
from problem1 import Tree
#-------------------------------------------------------------------------
'''
    Problem 2: Decision Tree (with continous attributes)
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''

#--------------------------
class Node:
    '''
        Decision Tree Node (with continous attributes)
        Inputs:
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar
            th: the threshold on the attribute, a float scalar.
            C1: the child node for values smaller than threshold
            C2: the child node for values larger than threshold
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X=None,Y=None, i=None,th=None,C1=None, C2=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.th = th
        self.C1= C1
        self.C2= C2
        self.isleaf = isleaf
        self.p = p


#-----------------------------------------------
class DT(Tree):
    '''
        Decision Tree (with contineous attributes)
        Hint: DT is a subclass of Tree class in problem1. So you can reuse and overwrite the code in problem 1.
    '''

    #--------------------------
    @staticmethod
    def cutting_points(X,Y):
        '''
            Find all possible cutting points in the continous attribute of X.
            (1) sort unique attribute values in X, like, x1, x2, ..., xn
            (2) consider splitting points of form (xi + x(i+1))/2
            (3) only consider splitting between instances of different classes
            (4) if there is no candidate cutting point above, use -inf as a default cutting point.
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                cp: the list of  potential cutting points, a float numpy vector.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        d = {}
        cp = []
        for i,x in enumerate(X):
            if x in d.keys():
                d[x].append(Y[i])
            else:
                d[x] = [Y[i]]
        ux = list(set(X))
        ux.sort()
        for i in xrange(len(ux)-1):
            x_1 = float(ux[i])
            x_2 = float(ux[i+1])
            for y_1 in set(d[x_1]):
                for y_2 in set(d[x_2]):
                    if y_1 != y_2:
                        cp.append((x_1+x_2)/2.0)
                        break
                if (x_1+x_2)/2.0 in cp:
                    break
        if cp == []:
            cp.append(float('-inf'))

        cp = np.array(cp)

        #########################################
        return cp

    #--------------------------
    @staticmethod
    def best_threshold(X,Y):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X.
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                th: the best threhold, a float scalar.
                g: the information gain by using the best threhold, a float scalar.
            Hint: you can reuse your code in problem 1.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        cp = DT.cutting_points(X,Y)
        ig = []
        for p in cp:
            newX = X.copy()
            for i,x in enumerate(newX):
                if x < p:
                    newX[i] = 0
                else:
                    newX[i] = 1
            ig.append(Tree.information_gain(Y,newX))

        g = max(ig)
        th = cp[ig.index(g)]

        if th == float('-inf'):
            g = -1

        #########################################
        return th,g


    #--------------------------
    def best_attribute(self,X,Y):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float).
            Here we use information gain to evaluate the attributes.
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        ag = []
        ath = []
        for row in X:
            th_n, g_n = self.best_threshold(row,Y)
            ath.append(th_n)
            ag.append(g_n)

        i = ag.index(max(ag))
        th = ath[i]

        #########################################
        return i, th




    #--------------------------
    @staticmethod
    def split(X,Y,i,th):
        '''
            Split the node based upon the i-th attribute and its threshold.
            (1) split the matrix X based upon the values in i-th attribute and threshold
            (2) split the labels Y
            (3) build children nodes by assigning a submatrix of X and Y to each node

            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C1: the child node for values smaller than threshold
                C2: the child node for values larger than (or equal to) threshold
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        ind1 = []
        ind2 = []
        for j,x in enumerate(X[i,:]):
            if x < th:
                ind1.append(j)
            else:
                ind2.append(j)
        C1 = Node(X = X[:,ind1],Y = Y[ind1])
        C2 = Node(X = X[:,ind2],Y = Y[ind2])

        #########################################
        return C1, C2



    #--------------------------
    def build_tree(self, t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C1: the child node for values smaller than threshold
                t.C2: the child node for values larger than (or equal to) threshold
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        # if Condition 1 or 2 holds, stop recursion
        t.p = Tree.most_common(t.Y)
        if Tree.stop1(t.Y) or Tree.stop2(t.X):
            t.isleaf = True
            return

        # find the best attribute to split
        t.i,t.th = self.best_attribute(t.X,t.Y)
        t.C1,t.C2 = self.split(t.X,t.Y,t.i,t.th)

        # recursively build subtree on each child node
        for child in [t.C1,t.C2]:
            self.build_tree(child)

        #########################################

    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively.
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        if not t.isleaf:
            sa = x[t.i]
            if sa < t.th:
                y = DT.inference(t.C1,x)
            else:
                y = DT.inference(t.C2,x)
        else:
            y = t.p

        #########################################
        return y


    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset.
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Y = []
        for i in xrange(X.shape[1]):
            Y.append(DT.inference(t,X[:,i]))
        Y = np.array(Y)
        #########################################
        return Y



    #--------------------------
    def train(self, X, Y):
        '''
            Given a training set, train a decision tree.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = Node(X = X, Y = Y)
        self.build_tree(t)


        #########################################
        return t


    #--------------------------
    @staticmethod
    def load_dataset(filename='data2.csv'):
        '''
            Load dataset 2 from the CSV file: data2.csv.
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




