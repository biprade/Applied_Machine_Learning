import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import train_test_split
import ROC_Curve
data=pd.read_csv(open('train_Complete.csv'))
target=data['TripType']
del data['TripType']
X, y =data,target
X_features=150
ch2 = SelectKBest(chi2, k=X_features)
X = ch2.fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

a= ROC_Curve.getROCScore(X_train,y_train,X_test,y_test,'Bagging')
print a[0],a[1]

