from pylab import *
from sklearn import preprocessing
import ml_algorithms, tools

feats=genfromtxt('sncosmo_des_fit.txt', dtype='str', comments='#')
#Features
f=array(feats[:, 5:10], dtype='float')
types=array(feats[:, 1], dtype='int')

X=f[types!=-9, :]
Y=types[types!=-9]

X_scaled = X
#X_scaled = preprocessing.scale(X)

Y[(Y==21) | (Y==22) | (Y==23)]=2
Y[(Y==32) | (Y==33)]=3


n=len(X[:, 0])
X_train=X_scaled[:n/2, :]
Y_train=Y[:n/2]

X_test=X_scaled[n/2:, :]
Y_test=Y[n/2:]

tools.run_ml(X_train, Y_train, X_test, Y_test, run_svm=False)
