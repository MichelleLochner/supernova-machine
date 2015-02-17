from pylab import *
from sklearn import preprocessing
import max_ml_algorithms as ml_algorithms, max_tools as tools

feats=genfromtxt('sncosmo_des_fit.txt', dtype='str', comments='#')

#Features
f=array(feats[:, 5:10], dtype='float')
types=array(feats[:, 1], dtype='int')

X=f[types!=-9, :]
Y=types[types!=-9]

#X_scaled = X
X_scaled = preprocessing.scale(X)

#Y[(Y==21) | (Y==22) | (Y==23)]=2
#Y[(Y==32) | (Y==33)]=3

for index in range(len(Y)):
    if Y[index]==21 or Y[index]==22 or Y[index]==23:
        Y[index] = 2
    elif Y[index]==32 or Y[index]==33:
        Y[index]=3


n=len(X[:, 0])
#split X and Y in half to get training and testing sets
X_train=X_scaled[:int(round(n/2.0)), :]
Y_train=Y[:int(round(n/2.0))]

X_test=X_scaled[int(round(n/2.0)):, :]
Y_test=Y[int(round(n/2.0)):]

tools.run_ml(X_train, Y_train, X_test, Y_test, run_svm=True)
