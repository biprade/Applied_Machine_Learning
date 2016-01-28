import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import KFold
from sklearn.externals import joblib
from random import randint
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn import tree
from sklearn.svm import LinearSVC
from ROC_Curve import getROCScore
import random

f = open("train_Complete.csv")

data = pd.read_csv(f)

target = data['TripType']
del data['TripType']

X, y = data, target

ch2 = SelectKBest(chi2, k=130)
X = ch2.fit_transform(X, y)

# Splitting the data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

kf = KFold(n=70686, n_folds=10, shuffle=False, random_state=42)
i=0
count = 0
max_score = 0
best_model = -1
best_CValue = 0
C_Value=[0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000,10000,100000]
total_roc=0
for train_index, test_index in kf:
    count +=1
    C = C_Value[i]
    i+=1

    X_train_fold, X_test_fold, y_train_fold, y_test_fold = X[train_index], X[test_index], y[train_index], y[test_index]
    
    result_ROC =  getROCScore(X_train_fold, y_train_fold, X_test_fold, y_test_fold, "LinearSVC", Cvalue=100000)

    roc_Micro_score = result_ROC[0]
    print("ROC on Validation set ",roc_Micro_score)
    total_roc+=roc_Micro_score
print("Average ROC Score is (Validation ROC) : ",total_roc/10)
result_ROC =getROCScore(X_train, y_train, X_test, y_test, "LinearSVC", Cvalue=100000)
print("ROC score on test set is ",result_ROC[0])
