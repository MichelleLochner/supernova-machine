from pylab import *
from sklearn import *
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from scipy.integrate import trapz
from scipy.stats.mstats import mode
import time
import numpy as np
import pybrain as pb
from pybrain.datasets import ClassificationDataSet
from pybrain.structure import LinearLayer,  SigmoidLayer,  SoftmaxLayer,  FullConnection
from pybrain.supervised.trainers import BackpropTrainer


#Make a roc curve to evaluate a classification routine
def roc(pr, Yt):
    probs=pr.copy()
    Y_test=Yt.copy()
    Y_test = Y_test.squeeze()
    if len(shape(pr))>1:
        probs_1=probs[:, 0]
    else:
        probs_1=probs
    threshold=linspace(0., 1., 500) #500 evenly spaced numbers between 0,1
    tpr=[0]*len(threshold)
    fpr=[0]*len(threshold)
    Y_test[Y_test==2]=0
    Y_test[Y_test==3]=0

    for i in range(len(threshold)):
        preds=zeros(len(Y_test))
        preds[probs_1>=threshold[i]]=1
        
        TP=sum((preds==1) & (Y_test==1))
        FP=sum((preds==1) & (Y_test==0))
        TN=sum((preds==0) & (Y_test==0))
        FN=sum((preds==0) & (Y_test==1))
        if TP==0:
            tpr[i]=0
        else:
            tpr[i]=TP/(float)(TP+FN)
            
        fpr[i]=FP/(float)(FP+TN)
    fpr=array(fpr)[::-1]
    tpr=array(tpr)[::-1]
    
    auc=trapz(tpr, fpr)
    return fpr, tpr, auc
    
    
#Calculate an F1 statistic
def F1(pr,  Yt):
    probs = pr.copy()
    Y_test = Yt.copy()
    
    threshold = np.arange(0, 1, 0.01)
    F1 = -299*ones(1)
    
    for T in range(len(threshold)):
        preds=2*ones(len(Y_test))
        preds[probs[:, 0]>=threshold[T]]=1
        
        TP=sum((preds==1) & (Y_test==1))
        FP=sum((preds==1) & (Y_test!=1))
        TN=sum((preds!=1) & (Y_test!=1))
        FN=sum((preds!=1) & (Y_test==1))
        
        if TP == 0 or FP == 0 or FN == 0:
            F1=np.append(F1, -9999)
        else:
            F1=np.append(F1, 2.0*TP/(2*TP+FP+FN))
        del TP,  FP,  FN,  TN,  preds
    
    F1 = np.delete(F1, 0)
    
    best_F1 = np.amax(F1)
    best_threshold_index = np.argmax(F1)
    best_threshold = threshold[best_threshold_index]
    #print("Best F1 threshold is: %s" %(best_threshold))
    
    return best_F1, best_threshold

#Calculate a Kessler FoM statistic
def FoM(pr,  Yt):
    probs = pr.copy()
    Y_test = Yt.copy()
    
    weight = 3.0

    threshold = np.arange(0, 1, 0.01)
    FoM = -299*ones(1)
    
    for T in range(len(threshold)):
        preds=2*ones(len(Y_test))
        preds[probs[:, 0]>=threshold[T]]=1
        
        TP=sum((preds==1) & (Y_test==1))
        FP=sum((preds==1) & (Y_test!=1))
        TN=sum((preds!=1) & (Y_test!=1))
        FN=sum((preds!=1) & (Y_test==1))
        
        if TP == 0 or FP == 0 or FN == 0:
            FoM=np.append(FoM, -9999)
        else:
            efficiency = float(TP)/(TP+FN)
            purity = float(TP)/(TP+weight*FP)
            FoM=np.append(FoM,efficiency*purity )
            del efficiency,  purity
        del TP,  FP,  FN,  TN,  preds
    
    FoM = np.delete(FoM, 0)

    best_FoM = np.amax(FoM)
    best_threshold_index = np.argmax(FoM)
    best_threshold = threshold[best_threshold_index]
    #print("Best FoM threshold is: %s" %(best_threshold))
    
    return best_FoM, best_threshold

#Calculate FoM by requiring a positive to have P[0] greater than a threshold 
#and be the largest class probability.
def alternative_FoM(probs, Y_test):
    max_probs = np.argmax(probs, axis=1)
    preds = (max_probs==0) & (probs[:, 0]>0.0)
    
    TP_list = (preds == True) & (Y_test == 1)
    FP_list = (preds == True) & (Y_test != 1)
    TN_list = (preds == False) & (Y_test !=1)
    FN_list = (preds == False) & (Y_test == 1)
    
    TP = sum(TP_list)
    FP = sum(FP_list)
    TN = sum(TN_list)
    FN = sum(FN_list)
    
    weight = 3.0
    
    print("alternative TP is: %s" %(TP))
    print("alternative FP is: %s" %(FP))
    print("alternative TN is: %s" %(TN))
    print("alternative FN is: %s" %(FN))
    
    efficiency = float(TP)/(TP+FN)
    purity = float(TP)/(TP+weight*FP)
    FoM = efficiency*purity
    
    return FoM

#SVM using the SVC routine 
def support_vm(X_train, Y_train, X_test, *args):
    
    a=time.time()
    
    clf=svm.SVC(kernel='linear', probability=True)
    
    f=clf.fit(X_train, Y_train)
    probs= clf.predict_proba(X_test)
    
    return probs



#SVM using the SVC routine with cubic kernel
def support_vm3(X_train, Y_train, X_test, Y_test, ):
    a=time.time()
    svr=svm.SVC(kernel='poly', degree = 3, probability=True)
    clf=svr
    print 'fitting now'
    f=clf.fit(X_train, Y_train)
    preds=clf.predict(X_test)
    probs= clf.predict_proba(X_test)
    print
    print 'Support vector machine'
    print 'Time taken', time.time()-a, 's'
    print 'Accuracy', sum(preds==Y_test)/(float)(len(preds))
    mismatched=preds[preds!=Y_test]
    
    print 'False Ia detection',  sum(mismatched==1)/(float)(sum(preds==1))

    return probs


#SVM using the SVC routine with radial basis function kernel
def support_vmRBF(X_train, Y_train, X_test, *args):

    clf=svm.SVC(kernel='rbf', probability = True, C = args[0], gamma = args[1])
    f=clf.fit(X_train, Y_train)
    probs= clf.predict_proba(X_test)

    return probs
    
    
#A bagged RF 
def forest(X_train, Y_train, X_test, *args):

    clf = RandomForestClassifier(n_estimators = args[0], criterion=args[1]) 
    clf.fit(X_train, Y_train)
    probs=clf.predict_proba(X_test)

    return probs



#A boosted RF
def boost_RF(X_train, Y_train, X_test, *args):
    
    classifier = AdaBoostClassifier(base_estimator = args[0], n_estimators = args[1])
    classifier.fit(X_train, Y_train)
    probs=classifier.predict_proba(X_test)    
    
    return probs


    

#KNN classifier with weights going as 1/r
def nearest_neighbours(X_train, Y_train, X_test, *args):

    clf=neighbors.KNeighborsClassifier(n_neighbors = args[0], weights = args[1])
    clf.fit(X_train, Y_train)
    probs=clf.predict_proba(X_test)
    
    return probs



#RNN classifier with weights going as 1/r
def radius_neighbours(X_train, Y_train, X_test, Y_test):
    
    try:
        a=time.time()
        radius=0.2      #Come back to this - need a rigorous approach
        clf=neighbors.RadiusNeighborsClassifier(radius, weights='distance')
        clf.fit(X_train, Y_train)
        preds=clf.predict(X_test)
        
        print
        print 'Radius nearest neighbours'
        print 'Time taken', time.time()-a, 's'
        print 'Accuracy', sum(preds==Y_test)/(float)(len(preds))
        mismatched=preds[preds!=Y_test]
        print 'False Ia detection',  sum(mismatched==1)/(float)(sum(preds==1))
        
        probs=clf.predict_proba(X_test)
        #fpr, tpr, auc=roc(probs, Y_test)
    except ValueError:
        print 'ValueError in RNN - probably due to no neighbours within radius'
        fpr,  tpr,  auc ,  probs,  Y_test= (None, None, None,  -9999,  None)
        
    #return fpr, tpr, auc
    return probs


#Naive Bayes classifier 
def bayes(X_train, Y_train, X_test, *args):

    clf = GaussianNB()
    f=clf.fit(X_train, Y_train)    
    probs=clf.predict_proba(X_test)

    return probs
    
    
    
#ANN classifier
def ANN(X_train, Y_train, X_test, Y_test, *args):
    Y_train_copy = Y_train.copy()
    Y_test_copy = Y_test.copy()

    #Convert class labels from 1,2,3 to 0,1,2 as _convertToOneOfMany requires this
    Y_train_copy[(Y_train_copy==1)]=0
    Y_train_copy[(Y_train_copy==2)]=1
    Y_train_copy[(Y_train_copy==3)]=2

    Y_test_copy[(Y_test_copy==1)]=0
    Y_test_copy[(Y_test_copy==2)]=1
    Y_test_copy[(Y_test_copy==3)]=2
    
    #Put all the data in datasets as required by pybrain
    Y_train_copy = np.expand_dims(Y_train_copy, axis=1)
    Y_test_copy = np.expand_dims(Y_test_copy, axis=1)
    traindata = ClassificationDataSet(X_train.shape[1], nb_classes = len(np.unique(Y_train_copy))) #Preallocate dataset
    traindata.setField('input', X_train) #Add named fields
    traindata.setField('target', Y_train_copy) 
    traindata._convertToOneOfMany() #Convert classes 0, 1, 2 to 001, 010, 100

    testdata = ClassificationDataSet(X_test.shape[1], nb_classes=len(np.unique(Y_test_copy)))
    testdata.setField('input', X_test)
    testdata.setField('target', Y_test_copy)
    testdata._convertToOneOfMany()

    #Create ANN with n_features inputs, n_classes outputs and HL_size nodes in hidden layer
    N = pb.FeedForwardNetwork()
    HL_size1 = X_train.shape[1]*2+2
    HL_size2 = X_train.shape[1]*2+2
    
    #Create layers and connections
    in_layer = LinearLayer(X_train.shape[1])
    hidden_layer1 = SigmoidLayer(HL_size1)
    hidden_layer2 = SigmoidLayer(HL_size2)
    out_layer = SoftmaxLayer(len(np.unique(Y_test_copy))) #Normalizes output so as to sum to 1

    in_to_hidden1 = FullConnection(in_layer, hidden_layer1)
    hidden1_to_hidden2 = FullConnection(hidden_layer1, hidden_layer2)
    hidden2_to_out = FullConnection(hidden_layer2, out_layer)

    #Connect them up
    N.addInputModule(in_layer)
    N.addModule(hidden_layer1)
    N.addModule(hidden_layer2)
    N.addOutputModule(out_layer)
    N.addConnection(in_to_hidden1)
    N.addConnection(hidden1_to_hidden2)
    N.addConnection(hidden2_to_out)

    N.sortModules()

    #Create the backpropagation object
    trainer = BackpropTrainer(N, dataset=traindata,  momentum=0.1, verbose=False, weightdecay=0.01)

    #Train the network on the data for some number of epochs
    for counter in np.arange(40):
        trainer.train()

    #Run the network on testing data
    #Get raw output scores for a given input
    probs = N.activate(X_test[0, :])
    probs = np.expand_dims(probs, axis=0)

    for counter in np.arange(X_test.shape[0]-1):
        next_probs = N.activate(X_test[counter+1, :])
        next_probs = np.expand_dims(next_probs, axis=0)
        probs = np.append(probs, next_probs, axis=0)
    
    return probs
    
    
    
def MCSprobs(probs1, probs2, probs3):
    mean_probs = (probs1+probs2+probs3)/3.0
    
    #Find examples where classifiers disagree
    preds1 = np.argmax(probs1, axis=1)
    preds2 = np.argmax(probs2, axis=1)
    preds3 = np.argmax(probs3, axis=1)
    
    bool_mask = (preds1!=preds2) | (preds2!=preds3) | (preds1!=preds3)
    indices = bool_mask == True
    
    return mean_probs,  indices
    
def MCSpreds(probs1, probs2, probs3, probs4):
    preds1 = np.argmax(probs1, axis=1)
    preds2 = np.argmax(probs2, axis=1)
    preds3 = np.argmax(probs3, axis=1)
    preds4 = np.argmax(probs4, axis=1)
    
    print("preds1 uniques are: %s" %(np.unique(preds1)))
    
    combined_preds = np.concatenate((preds1, preds2, preds3, preds4), axis=1)
    
    mode_preds = mode(combined_preds, axis=1)
    
    return mode_preds
    
    
"""
def SkyNet_wrapper(X_train, Y_train, X_test, params, *args):
    
    #Write training and testing data to files
    train_file = open(params['training_filepath'], 'w') #Must end _train.txt
    test_file = open(params['testing_filepath'], 'w') #Must end _test.txt

    value1 = (X_train.shape[1], '\n')
    str_val1 = str(value)
    value2 = (len(np.unique(Y_train)), '\n')
    str_val2 = str(value2)
    
    train_file.write(str_val) #1st line is number of features
    train_file.write(len(str_val2)) #2nd line is number of classes
    
    value1 = (X_test.shape[1], '\n')
    str_val1 = str(value1)
    value2 = (len(np.unique(Y_train)), '\n')
    str_val2 = str(value2)
    
    test_file.write(X_test.shape[1], '\n') #1st line is number of features
    test_file.write(len(np.unique(Y_train)), '\n') #2nd line is number of classes

    for rows in np.arange(X_train.shape[0]):
        for cols in np.arange(X_train.shape[1]):
            train_file.write(str(X_train[rows, cols]))
            train_file.write(',')
            train_file.write(str(X_train[rows, cols]))
            train_file.write(',')
            train_file.write(str(X_train[rows, cols]))
            train_file.write(',')
            train_file.write(str(X_train[rows, cols]))
            train_file.write(',')
            train_file.write(str(X_train[rows, cols]))
            train_file.write(',')
            train_file.write('\n')
            train_file.write(str(Y_train[rows]))
            train_file.write(',')
            train_file.write('\n')

    for rows in np.arange(X_test.shape[0]):
        for cols in np.arange(X_test.shape[1]):
            test_file.write(str(X_test[rows, cols]))
            test_file.write(',')
            test_file.write(str(X_test[rows, cols]))
            test_file.write(',')
            test_file.write(str(X_test[rows, cols]))
            test_file.write(',')
            test_file.write(str(X_test[rows, cols]))
            test_file.write(',')
            test_file.write(str(X_test[rows, cols]))
            test_file.write(',')
            test_file.write('\n')
            test_file.write(str(Y_test[rows]))
            test_file.write(',')
            test_file.write('\n')
    del rows

    train_file.close()
    test_file.close()

    #Write parameter file
    param_file = open(params['params_filepath'], 'w')
    
    param_file.write('#input_root \n')
    param_file.write(params['training_filepath'][:-8]) #Removes _train.txt 
    param_file.write('#output_root \n')
    param_file.write(params['output_filepath']) #Must be in the form /file/path/namestem
    param_file.write('#nhid \n')
    param_file.write(params[])
    
    #Run SkyNet
    
    #Read probabilities from output file
    
    
    return probs

"""
    
    
    
    
    
    
    
    
    
