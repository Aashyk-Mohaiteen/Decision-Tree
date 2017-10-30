import numpy as np
import pandas as pd
from sklearn.externals import joblib
#import train
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
balance_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data', sep= ',', header= None)
X = balance_data.values[:, 1:5]
Y = balance_data.values[:,0]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
clf_gini = joblib.load('trees.pkl')
#clf.predict(X[0:1])
y_pred = clf_gini.predict(X_test)
print(y_pred)
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)
