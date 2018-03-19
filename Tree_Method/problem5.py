import math
import numpy as np
from problem2 import DT,Node
#-------------------------------------------------------------------------
'''
    Problem 5: Boosting (on continous attributes).
               We will implement AdaBoost algorithm in this problem.
    You could test the correctness of your code by typing `nosetests -v test5.py` in the terminal.
'''

#-----------------------------------------------
class DS(DT):
    '''
        Decision Stump (with contineous attributes) for Boosting.
        Decision Stump is also called 1-level decision tree.
        Different from other decision trees, a decision stump can have at most one level of child nodes.
        In order to be used by boosting, here we assume that the data instances are weighted.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y, D):
        '''
            Compute the entropy of the weighted instances.
            Input:
                Y: a list of labels of the instances, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the entropy of the weighted samples, a float scalar
            Hint: you could use np.unique().
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        e = 0
        d = {}
        for y in np.unique(Y):
            d[y] = 0
        for i,y in enumerate(Y):
            d[y] += D[i]
        for p in d.values():
            try:
                e -= p * math.log(p,2)
            except ValueError:
                pass
        #########################################
        return e

    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X,D):
        '''
            Compute the conditional entropy of y given x on weighted instances
            Input:
                Y: a list of values, a numpy array of int/float/string values.
                X: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                ce: the weighted conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        ce = 0
        d = {}
        for x in np.unique(X):
            d[x] = []
        for i,x in enumerate(X):
            d[x].append(i)
        for (x,ind) in d.iteritems():
            Y_x = Y[ind]
            D_x = D[ind]
            p_x = float(sum(D_x))
            for i,w in enumerate(D[ind]):
                try:
                    D_x[i] = float(w)/p_x
                except ZeroDivisionError:
                    pass
            e_x = DS.entropy(Y_x,D_x)
            ce += p_x * e_x
        #########################################
        return ce

    #--------------------------
    @staticmethod
    def information_gain(Y,X,D):
        '''
            Compute the information gain of y after spliting over attribute x
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                g: the weighted information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g = DS.entropy(Y,D) - DS.conditional_entropy(Y,X,D)



        #########################################
        return g

    #--------------------------
    @staticmethod
    def best_threshold(X,Y,D):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. The data instances are weighted.
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
            Output:
                th: the best threhold, a float scalar.
                g: the weighted information gain by using the best threhold, a float scalar.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        cp = DT.cutting_points(X,Y)
        ig = []
        cp = list(cp)
        for p in cp:
            newX = X.copy()
            for i,x in enumerate(newX):
                if x < p:
                    newX[i] = 0
                else:
                    newX[i] = 1
            ig.append(DS.information_gain(Y,newX,D))

        g = max(ig)
        th = cp[ig.index(g)]

        if th == float('-inf'):
            g = -1

        #########################################
        return th,g

    #--------------------------
    def best_attribute(self,X,Y,D):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float). The data instances are weighted.
            Here we use information gain to evaluate the attributes.
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        ag = []
        ath = []
        for row in X:
            th_n, g_n = self.best_threshold(row,Y,D)
            ath.append(th_n)
            ag.append(g_n)

        i = ag.index(max(ag))
        th = ath[i]


        #########################################
        return i, th

    #--------------------------
    @staticmethod
    def most_common(Y,D):
        '''
            Get the most-common label from the list Y. The instances are weighted.
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
                D: the weights of instances, a numpy float vector of length n
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        d = {}
        for y in np.unique(Y):
            d[y] = 0
        for i,y in enumerate(Y):
            d[y] += D[i]
        y = sorted(d,key=lambda x:d[x])[-1]

        #########################################
        return y


    #--------------------------
    def build_tree(self, X,Y,D):
        '''
            build decision stump by overwritting the build_tree function in DT class.
            Instead of building tree nodes recursively in DT, here we only build at most one level of children nodes.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Return:
                t: the root node of the decision stump.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = Node(X,Y)
        # if Condition 1 or 2 holds, stop splitting
        t.p = self.most_common(t.Y,D)
        if DT.stop1(t.Y) or DT.stop2(t.X):
            t.isleaf = True
            return t
        # find the best attribute to split
        t.i,t.th = self.best_attribute(t.X,t.Y,D)
        # configure each child node
        ind1 = []
        ind2 = []
        for j,x in enumerate(X[t.i,:]):
            if x < t.th:
                ind1.append(j)
            else:
                ind2.append(j)
        X1 = X[:,ind1]
        Y1 = Y[ind1]
        t.C1 = Node(X1,Y1,isleaf = True)
        D1 = D[ind1]
        s = float(sum(D1))
        for i,w in enumerate(D[ind1]):
            D1[i] = float(w)/s
        t.C1.p = self.most_common(Y1,D1)
        X2 = X[:,ind2]
        Y2 = Y[ind2]
        t.C2 = Node(X2,Y2,isleaf = True)
        D2 = D[ind2]
        s = float(sum(D2))
        for i,w in enumerate(D[ind2]):
            D2[i] = float(w)/s
        t.C2.p = self.most_common(Y2,D2)
        #########################################
        return t



#-----------------------------------------------
class AB(DS):
    '''
        AdaBoost algorithm (with contineous attributes).
    '''

    #--------------------------
    @staticmethod
    def weighted_error_rate(Y,Y_,D):
        '''
            Compute the weighted error rate of a decision on a dataset.
            Input:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the weighted error rate of the decision stump
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        e = 0
        for i,r in enumerate(Y==Y_):
            if not r:
                e += D[i]
        #########################################
        return e

    #--------------------------
    @staticmethod
    def compute_alpha(e):
        '''
            Compute the weight a decision stump based upon weighted error rate.
            Input:
                e: the weighted error rate of a decision stump
            Output:
                a: (alpha) the weight of the decision stump, a float scalar.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        if e == 0:
            a = 500
        elif e == 1:
            a = -500
        else:
            a = 1./2.*math.log((1-e)/e)
        #########################################
        return a

    #--------------------------
    @staticmethod
    def update_D(D,a,Y,Y_):
        '''
            update the weight the data instances
            Input:
                D: the current weights of instances, a numpy float vector of length n
                a: (alpha) the weight of the decision stump, a float scalar.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels by the decision stump, a numpy array of length n. Each element can be int/float/string.
            Output:
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        oD = D.copy()
        e = math.e
        for i,r in enumerate(Y == Y_):
            if r:
                if a != 0:
                    D[i] = oD[i]* (e**(-a))
            else:
                D[i] = oD[i]* (e**a)
        s = float(sum(D))
        for i,w in enumerate(D):
            D[i] = float(w)/s
      #########################################
        return D

    #--------------------------
    @staticmethod
    def step(X,Y,D):
        '''
            Compute one step of Boosting.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the current weights of instances, a numpy float vector of length n
            Output:
                t:  the root node of a decision stump trained in this step
                a: (alpha) the weight of the decision stump, a float scalar.
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        d = DS()
        t = d.build_tree(X,Y,D)
        Y_ = DT.predict(t,X)
        e = AB.weighted_error_rate(Y,Y_,D)
        a = AB.compute_alpha(e)
        D = AB.update_D(D,a,Y,Y_)
        #########################################
        return t,a,D


    #--------------------------
    @staticmethod
    def inference(x,T,A):
        '''
            Given a bagging ensemble of decision trees and one data instance, infer the label of the instance.
            Input:
                x: the attribute vector of a data instance, a numpy vectr of shape p.
                   Each attribute value can be int/float
                T:  the root nodes of decision stumps, a list of length n_tree.
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                y: the class label, a scalar of int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        ay =  []
        for t in T:
            ay.append(DS.inference(t,x))
        d ={}
        for item in np.unique(ay):
            d[item] = 0
        for i,item in enumerate(ay):
            d[item] += A[i]
        y = sorted(d,key=lambda x:d[x])[-1]
        #########################################
        return y


    #--------------------------
    @staticmethod
    def predict(X,T,A):
        '''
            Given an AdaBoost and a dataset, predict the labels on the dataset.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                T:  the root nodes of decision stumps, a list of length n_tree.
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        n = X.shape[1]
        Y = []
        for col in range(n):
            x = X[:,col]
            Y.append(AB.inference(x,T,A))
        Y = np.array(Y)
        #########################################
        return Y


    #--------------------------
    @staticmethod
    def train(X,Y,n_tree=10):
        '''
            train adaboost.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                n_tree: the number of trees in the ensemble, an integer scalar
            Output:
                T:  the root nodes of decision stumps, a list of length n_tree.
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        T = []
        A = []
        # initialize weight as 1/n
        n = X.shape[1]
        D = np.ones(n)/float(n)
        # iteratively build decision stumps
        for tree in range(n_tree):
            t,a,D = AB.step(X,Y,D)
            T.append(t)
            A.append(a)
        A = np.array(A)
        #########################################
        return T, A





