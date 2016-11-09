import pandas as pd
from sklearn import datasets
from sklearn import tree
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import pdb

def splitData(data_X, data_Y, test_percent):
    num_samps = len(data_Y)
    all_data = pd.concat([data_Y, data_X], axis=1, join_axes=[data_Y.index])
    all_data = shuffle(all_data)
    
    testX = all_data[:round(test_percent*num_samps),1:]
    testY = all_data[:round(test_percent*num_samps),0]
    trainX = all_data[round(test_percent*num_samps):,1:]
    trainY = all_data[round(test_percent*num_samps):,0]
    
    return trainX, trainY, testX, testY

'''
Decision tree classification example
    See URL below to plot decision surface:
    http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html#sphx-glr-
        auto-examples-tree-plot-iris-py
'''
iris_X = pd.DataFrame(datasets.load_iris().data)
iris_Y = pd.DataFrame(datasets.load_iris().target)
iris_X.head(10)
iris_Y.head(10)

[trainX, trainY, testX, testY] = splitData(iris_X, iris_Y, 0.2)

clf = tree.DecisionTreeClassifier().fit(trainX, trainY)
iris_pred = clf.predict(testX)

plt.figure()
plt.plot(list(range(1,len(iris_pred)+1)),iris_pred,'-r')
plt.scatter(list(range(1,len(iris_pred)+1)),testY)
plt.legend(['Predicitons','Actual Values'])


'''
Decision tree regression example
    See URL below for example of fitting a sine wave:
    http://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
        #sphx-glr-auto-examples-tree-plot-tree-regression-py
'''
boston_X = pd.DataFrame(datasets.load_boston().data)
boston_Y = pd.DataFrame(datasets.load_boston().target)
boston_X.head(10)
boston_Y.head(10)

clf1 = tree.DecisionTreeRegressor(max_depth=5).fit(boston_X, boston_Y)
clf2 = tree.DecisionTreeRegressor(max_depth=10).fit(boston_X, boston_Y)
boston_pred = clf1.predict(boston_X)

plt.figure()
plt.scatter(boston_X[5],boston_Y,c='red')
plt.scatter(boston_X[5],boston_pred)
plt.legend(['Actual Values','Predicitons'])



