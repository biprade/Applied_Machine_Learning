from sklearn import ensemble
import pandas as pd
from sklearn.metrics import accuracy_score

train_data=pd.read_csv(open('train_final.csv'))
train_target=train_data['TripType']
del train_data['TripType']

clf=ensemble.GradientBoostingClassifier(max_depth=1)
model=clf.fit(train_data,train_target)

test_data=pd.read_csv(open('test_final.csv'))
test_target=test_data['TripType']
del test_data['TripType']

predicted=model.predict(test_data)

print accuracy_score(test_target,predicted)