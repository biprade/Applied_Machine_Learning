from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.metrics import log_loss
import numpy as np
import ROC_Curve
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
data=pd.read_csv(open('/users/biprade/downloads/Project/train_Complete.csv'))
target=data['TripType']
del data['TripType']
X, y =data,target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)


a= ROC_Curve.getROCScore(X_train,y_train,X_test,y_test,'LinearSVC',Cvalue=1)
print a[0],a[1]
