'''
This class was used to get a threshold for selecting number of features to be used.
We increased the number of features and recorded the AUCROC score.
'''


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn import tree
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from random import randint
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from ROC_Curve import getROCScore
f = open("train_Complete.csv")

data = pd.read_csv(f)

target = data['TripType']
del data['TripType']

for X_features in range(1, 974, 50):
    X, y = data, target
    print("Feature Count::\t" +str(X_features))
    X_features=150
    ch2 = SelectKBest(chi2, k=X_features)
    X = ch2.fit_transform(X, y)

    # Splitting the data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

    # Decision Tree

    print("Decision Tree model")
    # d = clf.fit(X_train, y_train)
    print("starting predication")
    # Y_predicted = d.predict(X_test)
    print("Decision Tree Accuracy is")
    result_ROC = getROCScore(X_train, y_train, X_test, y_test, "DecisionTree", depth=10)
    print result_ROC[0]
    print("\n")

    #Linear Regression
    # regr=OneVsRestClassifier(linear_model.LogisticRegression(C=10000))
    print("Logistic Regression model")
    # d = regr.fit(X_train, y_train)
    print("starting predication")
    # Y_predicted = d.predict(X_test)

    print("Logistic Regression model Accuracy is")
    result_ROC = getROCScore(X_train, y_train, X_test, y_test, "LogisticRegression", Cvalue=10000)
    print result_ROC[0]
    print("\n")

    #SVM
    # svm = OneVsRestClassifier(LinearSVC(C=100000))
    print("SVM model")
    # d = svm.fit(X_train, y_train)
    print("starting predication")
    # Y_predicted = d.predict(X_test)
    print("SVM model Accuracy is")
    result_ROC = getROCScore(X_train, y_train, X_test, y_test, "LinearSVC", Cvalue=100000)
    print result_ROC[0]
    print("\n")


    #Naive Bayes
    print("Naive Bayes")
    # d = svm.fit(X_train, y_train)
    print("starting predication")
    # Y_predicted = d.predict(X_test)
    print("Naive Bayes model Accuracy is")
    result_ROC = getROCScore(X_train, y_train, X_test, y_test, "NaiveBayes", alphaValue=0.6)
    print result_ROC[0]
    print("\n")
    print("-------------------------------------------------------------------------------------------------------------")