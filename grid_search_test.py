#Investigating what on earth grid_search is actually doing

import numpy
import time
import max_ml_algorithms as ml_algorithms
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier
from sklearn.svm import SVC
from pylab import *
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


feats=genfromtxt('sncosmo_des_fit_emcee.txt', dtype='str', comments='#')

#Get features from classified samples
f=array(feats[:, 5:10], dtype='float')
types=array(feats[:, 1], dtype='int')

X=f[types!=-9, :]
Y=types[types!=-9]

X_scaled = preprocessing.scale(X)
print("mean of values in X_scaled is: ", numpy.mean(X_scaled))
print("std of values in X_scaled is: ",  numpy.std(X_scaled))
print("range of values in X_scaled is: ",  numpy.ptp(X_scaled))

#grid_search can only deal with binary classification, and needs class labels 
#either {0,1} or {-1,1}. Note this changes the order of 1A and 'not 1A' classes
Y[(Y!=1)]=0

#Split dataset randomly into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.5, random_state=0)

#Check there are only two classes
print("Unique classes are: ",  numpy.unique(Y))

#Set the parameters for optimisation and the optimisation metric
tuned_parameters = [{'kernel':['rbf'], 'C':[0.5, 1, 2] }]
score = 'roc_auc'

print("# Tuning hyper-parameters for %s \n" % score)

#Create the grid search classification object and fit it to data
clf = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, scoring=score)
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
dec_values = clf.best_estimator_.decision_function(X_test)

abs_dec_values = numpy.absolute(dec_values)
max_val = numpy.amax(abs_dec_values)
new_dec_values = dec_values/(2.0*max_val) + 0.5 #Normalise to -0.5 to 0.5

#Select the 2nd column of probs for the roc calculation as these are the probs 
#of being class 1 (i.e. being a 1A SN), for comparison with grid_search's own score values
#fpr,  tpr,  auc = ml_algorithms.roc(probs[:, 1], y_test)
fpr,  tpr,  auc = ml_algorithms.roc(dec_values, y_test)
print("My area under the curve from dec_vals is: %s \n"  %(auc))
    
print(classification_report(y_true, y_pred))
print()

#Use the best estimator
clf.best_estimator_.fit(X_train, y_train)
bestprobs = clf.best_estimator_.predict_proba(X_test)

print("The first ten values in bestprobs: \n")
print(bestprobs[:10])
print()
print("The first ten values in y_test are: \n")
print(y_test[:10])

fpr2,  tpr2,  auc2 = ml_algorithms.roc(bestprobs[:, 1], y_test)

print("My best estimator AUC from preds is %s " %(auc2))
#print("Best metric is: %s " %(clf.best_estimator_.metric))
print("Best C is: %s " %(clf.best_estimator_.C))
#print("Best degree is: %s " %(clf.best_estimator_.degree))
#print("Best gamma is %s " %(clf.best_estimator_.gamma))

weight = 3.0

preds=2*ones(len(y_test))
preds = numpy.argmax(bestprobs, 1)

TP=sum((preds==1) & (y_test==1))
FP=sum((preds==1) & (y_test!=1))
TN=sum((preds!=1) & (y_test!=1))
FN=sum((preds!=1) & (y_test==1))
    
if TP == 0 or FP == 0 or FN == 0:
    FoM=np.append(FoM, -9999)
else:
    efficiency = float(TP)/(TP+FN)
    purity = float(TP)/(TP+weight*FP)
    FoM=efficiency*purity 

print("Efficiency is %s" %(efficiency))
print("Purity is %s" %(purity))
print("FoM is %s" %(FoM))
    
    
    
    
    
    
    
    
    
    
