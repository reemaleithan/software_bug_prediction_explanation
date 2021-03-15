import numpy as np
import csv as csv
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import metrics

input_file_training = "./train.csv"
input_file_test = "./test.csv"

# load the training data as a matrix
dataset = pd.read_csv(input_file_training, header=0)

# separate the data from the target attributes
train_data = dataset.drop('label', axis=1)

# remove unnecessary features
#train_data = train_data.drop('File', axis=1)


# the lables of training data. `label` is the title of the  last column in your CSV files
train_target = dataset.label 

# load the testing data
dataset2 = pd.read_csv(input_file_test, header=0)

# separate the data from the target attributes
test_data = dataset2.drop('label', axis=1)

# remove unnecessary features
# test_data = test_data.drop('File', axis=1)

# the lables of test data
test_target = dataset2.label

#print(test_target)


gnb = GaussianNB()
gnb_pred = gnb.fit(train_data, train_target).predict(test_data)

dt = DecisionTreeClassifier(random_state=0, max_depth=2)
dt_pred = dt.fit(train_data, train_target).predict(test_data)


rf = RandomForestClassifier(n_estimators = 100)
rf_pred = rf.fit(train_data, train_target).predict(test_data)

lr = LogisticRegression(random_state=0)
lr_pred = lr.fit(train_data, train_target).predict(test_data)
fpr, tpr, thresholds = metrics.roc_curve(test_target,rf_pred , pos_label=1)
print("Random Forest")
auc = metrics.auc(fpr, tpr)
print(auc)
print ('\n')

fpr, tpr, thresholds = metrics.roc_curve(test_target,gnb_pred , pos_label=1)
print("Naive Bayes")
auc = metrics.auc(fpr, tpr)
print(auc)
print('\n')

fpr, tpr, thresholds = metrics.roc_curve(test_target,lr_pred , pos_label=1)
print("Logistic Regression")
auc = metrics.auc(fpr, tpr)
print(auc)
print('\n')

fpr, tpr, thresholds = metrics.roc_curve(test_target,dt_pred , pos_label=1)
print("Decision Tree")
auc = metrics.auc(fpr, tpr)
print(auc)
print('\n')

'''
print("GNB")
print(classification_report(test_target, gnb_pred, labels=[0,1]))
print("DT")
print(classification_report(test_target, dt_pred, labels=[0,1]))
print("rf")
print(classification_report(test_target, rf_pred, labels=[0,1]))
print(auc)
print("lr")
print(classification_report(test_target, lr_pred, labels=[0,1]))
'''



#print(classification_report(test_target, rf_pred, labels=[0,1]))
