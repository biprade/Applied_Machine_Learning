import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier

def getROCScore(X_train, y_train, X_test, y_test, classifierName, depth=None, Cvalue=1,alphaValue=0.0):



# Binarize the output
    y_train = label_binarize(y_train, classes=[3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 999])
    n_classes = y_train.shape[1]
    y_test = label_binarize(y_test, classes=[3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 999])



# Learn to predict each class against the other
    if classifierName=='DecisionTree':
        classifier=OneVsRestClassifier(tree.DecisionTreeClassifier(max_depth=depth))
    elif classifierName=='LogisticRegression':
        classifier = OneVsRestClassifier(linear_model.LogisticRegression(C=Cvalue))
    elif classifierName=='LinearSVC':
        classifier= OneVsRestClassifier(LinearSVC(C=Cvalue))
    elif classifierName=='NaiveBayes':
        classifier= OneVsRestClassifier(MultinomialNB(alpha=alphaValue))
    elif classifierName=='Bagging':
        estimator= tree.DecisionTreeClassifier()
        classifier=OneVsRestClassifier(BaggingClassifier(base_estimator=estimator))

    
    y_score = classifier.fit(X_train, y_train).predict(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
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

    return (roc_auc["micro"],roc_auc["macro"],classifier)







