import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import KFold
from sklearn.externals import joblib
from ROC_Curve import getROCScore
from sklearn import naive_bayes

f = open("train_Complete.csv")

data = pd.read_csv(f)

target = data['TripType']
del data['TripType']

X, y = data, target

ch2 = SelectKBest(chi2, k=150)
X = ch2.fit_transform(X, y)

# Splitting the data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

kf = KFold(n=len(X_train), n_folds=10, shuffle=False, random_state=42)

count = 0
max_score = 0
best_model = -1
best_alphaValue = 0
alpha_Value=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1]
i=0
total_roc=0
for train_index, test_index in kf:
    count +=1
    alpha = alpha_Value[i]
    i+=1

    X_train_fold, X_test_fold, y_train_fold, y_test_fold = X[train_index], X[test_index], y[train_index], y[test_index]
    
    result_ROC =  getROCScore(X_train_fold, y_train_fold, X_test_fold, y_test_fold, "NaiveBayes", alphaValue=0.6)

    roc_Micro_score = result_ROC[0]
    print("ROC on Validation set ",roc_Micro_score)
    total_roc+=roc_Micro_score
print("Average ROC Score is (Validation ROC) : ",total_roc/10)
result_ROC =getROCScore(X_train, y_train, X_test, y_test, "NaiveBayes", alphaValue=0.6)
print("ROC score on test set is ",result_ROC[0])
  