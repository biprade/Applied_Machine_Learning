from sklearn import ensemble
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.cross_validation import train_test_split
data=pd.read_csv(open('train_Complete.csv'))
target=data['TripType']
del data['TripType']
X, y =data,target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

clf=ensemble.AdaBoostClassifier(n_estimators=100)
model=clf.fit(X_train,y_train)


test_data=pd.read_csv(open('test_final.csv'))
test_target=test_data['TripType']
del test_data['TripType']

predicted=model.predict(X_test)

print accuracy_score(y_test,predicted)