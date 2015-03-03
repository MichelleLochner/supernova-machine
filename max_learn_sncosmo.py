from pylab import *
import numpy
from sklearn import preprocessing
import max_ml_algorithms as ml_algorithms, max_tools as tools
from sklearn.cross_validation import train_test_split

feats=genfromtxt('sncosmo_des_fit_emcee.txt', dtype='str', comments='#')

#Features
f=array(feats[:, 5:10], dtype='float')
types=array(feats[:, 1], dtype='int')

X=f[types!=-9, :]
Y=types[types!=-9]

#X_scaled = X
X_scaled = preprocessing.scale(X)

#Note - this catches all instances, not just the first because Y is an array
#and not a list
Y[(Y==21) | (Y==22) | (Y==23)]=2
Y[(Y==32) | (Y==33)]=3

"""
#split X and Y in half to get training and testing sets
n=len(X[:, 0])
X_train=X_scaled[:int(round(n/2.0)), :]
Y_train=Y[:int(round(n/2.0))]

X_test=X_scaled[int(round(n/2.0)):, :]
Y_test=Y[int(round(n/2.0)):]
"""

#Split dataset randomly into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.5, random_state=0)

tools.run_ml(X_train, Y_train, X_test, Y_test, run_svm=True)
