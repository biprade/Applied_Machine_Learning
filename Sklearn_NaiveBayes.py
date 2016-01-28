import pandas as pd

import ROC_Curve
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
data=pd.read_csv(open('/users/biprade/downloads/Project/train_Complete.csv'))
target=data['TripType']
del data['TripType']
X, y =data,target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

a= a= ROC_Curve.getROCScore(X_train,y_train,X_test,y_test,'NaiveBayes',alphaValue=0.6)
print a[0],a[1]
