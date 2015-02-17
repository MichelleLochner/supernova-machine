#Investigating what on earth grid_search is actually doing

import numpy
import time
import max_ml_algorithms as ml_algorithms
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from pylab import *
from sklearn import preprocessing


feats=genfromtxt('sncosmo_des_fit.txt', dtype='str', comments='#')

#Get features from classified samples
f=array(feats[:, 5:10], dtype='float')
types=array(feats[:, 1], dtype='int')

X=f[types!=-9, :]
Y=types[types!=-9]

X_scaled = preprocessing.scale(X)

#grid_search can only deal with binary classification, and needs class labels 
#either {0,1} or {-1,1}
for index in range(len(Y)):
    if Y[index]!=1:
        Y[index]=0

#Split dataset randomly into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.5, random_state=0)

#Check there are only two classes
print("Unique classes are: ",  numpy.unique(Y))


# Set the parameters for optimisation and the optimisation metric
#tuned_parameters = [{'criterion': ['gini'], 'n_estimators': [3, 500],
#                     'max_features': [2, 4]},
#                    {'criterion': ['entropy'], 'n_estimators': [3, 500], 'max_features':[2, 4]}]

tuned_parameters = [{'kernel':['poly'], 'degree':[2, 3, 4], 'C':[0.5, 1]}]

score = 'roc_auc'

start_time = time.time()

print("# Tuning hyper-parameters for %s \n" % score)

#Create the grid search classification object and fit it to data
clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring=score)
clf.fit(X_train, y_train)

print("Best parameter set found on development set: \n")
print(clf.best_estimator_)
print()
print("Grid score on development set: \n")

for params, mean_score, score in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, score.std() / 2, params))

print()

print("Detailed classification report: \n")
print("The model is trained on the full development set. \n")
print("The score are computed on the full evaluation set. \n")
    
y_true, y_pred = y_test, clf.predict(X_test)
probs= clf.predict_proba(X_test)

end_time = time.time()

print("Time taken was %s s \n" %(end_time-start_time))

#Select the 2nd column of probs for the roc calculation as these are the probs 
#of being class 1 (i.e. being a 1A SN), for comparison with grid_search's own score values
fpr,  tpr,  auc = ml_algorithms.roc(probs[:, 1], y_test)
print("My area under the curve is: %s \n"  %(auc))

#figure(figsize=(10, 10))
#C1='#a21d21' #dark red
#linew = 2.5
#plot(f1, t1, C1, lw=linew)
    
#show()
    
print(classification_report(y_true, y_pred))
print()

#Use the optimum parameter, instead of searching for optimum params each time
clf.best_estimator_.fit(X_train, y_train)
bestprobs = clf.best_estimator_.predict_proba(X_test)

print("The first ten values in bestprobs: \n")
print(bestprobs[:10])
print()

fpr2,  tpr2,  auc2 = ml_algorithms.roc(bestprobs[:, 1], y_test)

print("My best estimator AUC is %s \n" %(auc2))
    
    
    
    
    
    
    
    
    
    
    
    
