import pandas as pd
import numpy as np
from collections import Counter
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
data=pd.read_csv(open('train_Complete.csv'))
target=data['TripType']
del data['TripType']
X, y =data,target
X_features=150
ch2 = SelectKBest(chi2, k=X_features)
X = ch2.fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
y_test = label_binarize(y_test, classes=[3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 999])

def SVM(train_data,train_target,test_data):
    clf=OneVsRestClassifier(LinearSVC(C=100000)).fit(train_data, train_target).fit(train_data,train_target)
    predicted_output=clf.predict(test_data)
    return predicted_output

def Logistic(train_data,train_target,test_data):
    clf=OneVsRestClassifier(linear_model.LogisticRegression(C=10000)).fit(train_data,train_target)
    predicted_output=clf.predict(test_data)
    return predicted_output
def DecisionTrees(train_data,train_target,test_data):
    clf=OneVsRestClassifier(tree.DecisionTreeClassifier(max_depth=70)).fit(train_data,train_target)
    predicted_output=clf.predict(test_data)
    return predicted_output

def NaiveBayes(train_data,train_target,test_data):
    clf=OneVsRestClassifier(MultinomialNB(alpha=0.6)).fit(train_data,train_target)
    predicted_output=clf.predict(test_data)
    return predicted_output

output1=SVM(X_train,y_train,X_test)
output2=Logistic(X_train,y_train,X_test)
output3=DecisionTrees(X_train,y_train,X_test)
output4=NaiveBayes(X_train,y_train,X_test)

output=list()
y_score=list()
output_location={3:0, 4:1, 5:2, 6:3, 7:4, 8:5, 9:6, 12:7, 14:8, 15:9, 18:10, 19:11, 20:12, 21:13, 22:14, 23:15, 24:16, 25:17, 26:18, 27:19, 28:20, 29:21, 30:22, 31:23, 32:24, 33:25, 34:26, 35:27, 36:28, 37:29, 38:30, 39:31, 40:32, 41:33, 42:34, 43:35, 44:36, 999:37}
for i in range(0,len(X_test)):
    temp=[output1[i],output2[i],output3[i],output4[i]]

    output.append(temp)


for i in range(0,len(output)):
    most_common= Counter(output[i]).most_common(1)[0][0]
    temp=[0]*38
    temp[output_location[most_common]]=1
    y_score.append(temp)


n_classes=38
fpr = dict()
tpr = dict()
roc_auc = dict()
y_test=np.array(y_test.tolist())
y_score=np.array(y_score)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
print "RUC MICRO SCORE ",roc_auc["micro"]
print "RUC MACRO SCORE ",roc_auc["macro"]




