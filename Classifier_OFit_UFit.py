
'''
This class is used to plot the train and test log loss against varying training size
to detect overfit-underfit in the model
'''

from __future__ import division

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.multiclass import OneVsRestClassifier
from sklearn import tree
from sklearn.metrics import log_loss
from sklearn.cross_validation import KFold
from sklearn.preprocessing import label_binarize
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn import naive_bayes
f = open("train_Complete.csv")

data = pd.read_csv(f)
target = data['TripType']
del data['TripType']
X, y = data, target

ch2 = SelectKBest(chi2, k=150)
X = ch2.fit_transform(X, y)
size=40
classifier=OneVsRestClassifier(LinearSVC(C=5))

for i in range(1,11):
    s=size/100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=s, random_state=42)
    trainSetSize=len(X_train)
    print trainSetSize
    print X_train.shape
    kf = KFold(n=trainSetSize,shuffle=True, random_state=42)
    logLoss=0
    for train_index, test_index in kf:
        X_train_fold, X_test_fold, y_train_fold, y_test_fold = np.array(X_train)[train_index], np.array(X_train)[test_index], np.array(y_train)[train_index], np.array(y_train)[test_index]
        y_train_fold = label_binarize(np.array(y_train_fold), classes=[3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 999])
        y_test_fold = label_binarize(np.array(y_test_fold), classes=[3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 999])
        y_score = classifier.fit(X_train_fold, y_train_fold).decision_function(X_test_fold)
        logLoss+=log_loss(y_test_fold,y_score)



    print("Training Size :",len(X_train))
    print("Test Size :",len(X_test))
    print("Log Loss on Train Data : ",logLoss/3)
    y_train = label_binarize(y_train, classes=[3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 999])
    y_test = label_binarize(y_test, classes=[3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 999])
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    print("Log Loss on Test Data : ",log_loss(y_test,y_score))
    print("-"*100)
    size=size-2

