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
    """
    Produce the false positive rate and true positive rate required to plot
    a ROC curve, and the area under that curve.
    
    INPUTS:
    pr - An array of probability scores, of size (N_samples,)
    Yt - An array of class labels, of size (N_samples,)
    
    OUTPUTS:
    fpr - An array containing the false positive rate at each probability threshold
    tpr - An array containing the true positive rate at each probability threshold
    auc - The area under the ROC curve
    """
    
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
    
    
def F1(pr,  Yt):
    """
    Calculate an F1 score for many probability threshold increments 
    and select the best one. 
    
    F1 is defined as:
    F1 = 2*TP/(2*TP+FP+FN)
    
    INPUTS:
    pr - An array containing probability scores, of size (N_samples,)
    Yt - An array containing class labels, of size (N_samples,)
    
    OUTPUTS:
    best_F1 - The largest F1 value
    best_threshold - The probability threshold corresponding to best_F1
    """
    
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
    
    return best_F1, best_threshold

def FoM(pr,  Yt):
    """
    Calculate a Kessler FoM for many probability threshold increments
    and select the largest one.
    
    FoM is defined as:
    FoM = TP^2/((TP+FN)(TP+3*FP))
    
    INPUTS:
    pr - An array of probability scores, of size (N_samples,)
    Yt - An array of class labels, of size (N_samples,)
    
    OUTPUTS:
    best_FoM - The largest FoM value
    best_threshold - The probability threshold corresponding
                                to best_FoM
    """
    
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
    
    return best_FoM, best_threshold


#Calculate FoM by requiring a positive to have P[0] greater than a threshold 
#and be the largest class probability.
#def alternative_FoM(probs, Y_test):
#    max_probs = np.argmax(probs, axis=1)
#    preds = (max_probs==0) & (probs[:, 0]>0.0)
    
#    TP_list = (preds == True) & (Y_test == 1)
#    FP_list = (preds == True) & (Y_test != 1)
#    TN_list = (preds == False) & (Y_test !=1)
#    FN_list = (preds == False) & (Y_test == 1)
    
#    TP = sum(TP_list)
#    FP = sum(FP_list)
#    TN = sum(TN_list)
#    FN = sum(FN_list)
    
#    weight = 3.0
    
#    print("alternative TP is: %s" %(TP))
#    print("alternative FP is: %s" %(FP))
#    print("alternative TN is: %s" %(TN))
#    print("alternative FN is: %s" %(FN))
    
#    efficiency = float(TP)/(TP+FN)
#    purity = float(TP)/(TP+weight*FP)
#    FoM = efficiency*purity
    
#    return FoM



def support_vm(X_train, Y_train, X_test, *args):
    """
    Implements a linear Support Vector Machine classifier.
    
    INPUTS:
    X_train - An array containing the features of the training
                set, of size (N_samples, N_features)
    Y_train - An array containing the class labels of the training
                set, of size (N_samples,)
    X_test - An array containing the features of the testing
                set, of size (N_samples,)
    
    OUTPUTS:
    probs - An array containing the probabilities for each class
                for each member of the testing set, of size 
                (N_samples, N_classes)
    """
    
    a=time.time()
    
    #Create the SVM
    clf=svm.SVC(kernel='linear', probability=True)
    
    #Train the SVM and use it to classify the testing set
    f=clf.fit(X_train, Y_train)
    probs=clf.predict_proba(X_test)
    
    return probs


def support_vm3(X_train, Y_train, X_test, Y_test, ):
    """
    Implements a Support Vector Machine classifier with cubic kernel.
    
    INPUTS:
    X_train - An array containing the features of the training set, of 
                size (N_samples, N_features)
    Y_train - An array containing the class labels of the training set, 
                of size (N_samples,)
    X_test - An array containing the features of the testing set, of
                size (N_samples, N_features)
    Y_test - An array containing the class labels of the testing set, of
                size (N_samples,)
    
    OUTPUTS:
    probs - An array containing the probabilies for each class for each 
                member of the testing set, of size (N_samples, N_classes)
    """
    
    a=time.time()
    
    #Create the classifier
    clf=svm.SVC(kernel='poly', degree = 3, probability=True)
 
    #Train the classifier, and use it to classify the testing set
    f=clf.fit(X_train, Y_train)
    probs= clf.predict_proba(X_test)
 
    return probs

def support_vmRBF(X_train, Y_train, X_test, *args):
    """
    Implements a Support Vector Machine classifier with radial basis function kernel. The 
    kernel function is defined as:
    
    K = exp(-gamma*|x-x'|^2)
    
    INPUTS:
    X_train - An array containing the features of the training set, of size (N_samples, N_features)
    Y_train - An array containing the class labels of the training set, of size (N_samples,)
    X_test - An array containing the features of the testing set, of size (N_samples, N_features)
    *args - The parameters taken by the SVM. C is the cost of misclassification, gamma is a 
                parameter of the kernel.
    
    OUTPUTS:
    probs - An array containing the probabilities for each class for each member of the 
                testing set, of size (N_samples, N_classes)
    """

    #Create the classifier
    clf=svm.SVC(kernel='rbf', probability = True, C = args[0], gamma = args[1])
    
    #Train the classifier, and use it to classify the testing set
    f=clf.fit(X_train, Y_train)
    probs= clf.predict_proba(X_test)

    return probs
    
    
def forest(X_train, Y_train, X_test, *args):
    """
    Implements a Bagged Random Forest of Decision Trees.
    
    INPUTS:
    X_train - An array containing the features of the training set, of size (N_samples, N_features)
    Y_train - An array containing the class labels of the training set, of size (N_samples,)
    X_test - An array containing the features of the test set, of size (N_samples, N_features)
    *args - The parameters of the classifier. n_estimators is the number of trees in the forest
                and criterion is whether data splits at each tree node are done to extremize 
                information entropy or the gini score.

    OUTPUTS:
    probs - An array containing the probabilites for each class for each member of the testing
                set, of size (N_samples, N_classes)
    """
    #Create the classifier
    clf = RandomForestClassifier(n_estimators = args[0], criterion=args[1]) 
    
    #Train the classifier, and use it to classify the testing set
    clf.fit(X_train, Y_train)
    probs=clf.predict_proba(X_test)

    return probs



def boost_RF(X_train, Y_train, X_test, *args):
    """
    Implements a boosted ensemble of the base_estimator.
    
    INPUTS:
    X_train - An array containing the features of the training set, of size (N_samples, N_features)
    Y_train - An array containing the class labels of the training set, of size (N_samples,)
    X_test - An array containing the features of the testing set, of size (N_samples, N_features)
    *args - The parameters of the classifier. base_estimator is either a decision tree or a bagged
                random forest of decision trees, n_estimators is the number of these estimators in the
                ensemble.
    
    OUTPUTS:
    probs - An array containing the probabilities for each class for each member of the testing set,
                of size (N_samples, N_classes)
    """
    
    #Create the classifier
    classifier = AdaBoostClassifier(base_estimator = args[0], n_estimators = args[1])
    
    #Train the classifier, and use it to classify the testing set
    classifier.fit(X_train, Y_train)
    probs=classifier.predict_proba(X_test)    
    
    return probs


    
def nearest_neighbours(X_train, Y_train, X_test, *args):
    """
    Implements a K-nearest neighbours classifier.
    
    INPUTS:
    X_train - An array containing the features of the training set, of size (N_samples, N_features)
    Y_train - An array containing the class labels of the training set, of size (N_samples,)
    X_test - An array containing the features of the testing set, of size (N_samples, N_features)
    *args - The parameters of the KNN. n_neighbours is the number of neighbours used to classify
                new data points (AKA K), and weights is how those neighbours are weighted (e.g. their
                contribution is ~1/r)
    
    OUTPUTS:
    probs - An array containing the probabilities for each class for each member of the testing set, 
                of size (N_samples, N_classes)
    """
    
    #Create the classifier
    clf=neighbors.KNeighborsClassifier(n_neighbors = args[0], weights = args[1])
    
    #Train the classifier, and use it to classify the testing set
    clf.fit(X_train, Y_train)
    probs=clf.predict_proba(X_test)
    
    return probs



#RNN classifier with weights going as 1/r
#CURRENTLY NOT USED
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



def bayes(X_train, Y_train, X_test, *args):
    """
    Implements a Naive Bayes classifier.
    
    INPUTS:
    X_train - An array containing the features of the training set, of size (N_samples, N_features)
    Y_train - An array containing the class labels of the training set, of size (N_samples,)
    X_test - An array containing the features of the testing set, of size (N_samples, N_features)
    *args - Currently not used as Naive Bayes has no parameters to optimise
    
    OUTPUTS:
    probs - An array containing the probabilities for each class for each member of the testing
                set, of size (N_samples, N_classes)
    """

    #Create the classifier
    clf = GaussianNB()
    
    #Train the classifier, and use it to classify the testing set
    f=clf.fit(X_train, Y_train)    
    probs=clf.predict_proba(X_test)

    return probs
    
    
    
#ANN classifier
def ANN(X_train, Y_train, X_test, Y_test, *args):
    """
    An Artificial Neural Network, based on the python library pybrain. In the future this function
    should be modified to use the SkyNet ANN code instead.
    
    INPUTS:
    X_train - An array containing the features of the training set, of size (N_samples, N_features)
    Y_train - An array containing the class labels of the training set, of size (N_samples,)
    X_test - An array containing the features of the testeing set, of size (N_samples, N_features)
    Y_test - An array containing the class labels of the testing set, of size (N_samples)
    *args - Currently unused. In the future could specify the network architecture and activation
                functions at each node.
    
    OUTPUTS:
    probs - an array containing the probabilities for each class for each member of the testing set,
                of size (N_samples, N_classes)
    """
    
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

    #Create ANN with n_features inputs, n_classes outputs and HL_size nodes in hidden layers
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
    probs = N.activate(X_test[0, :])
    probs = np.expand_dims(probs, axis=0)

    for counter in np.arange(X_test.shape[0]-1):
        next_probs = N.activate(X_test[counter+1, :])
        next_probs = np.expand_dims(next_probs, axis=0)
        probs = np.append(probs, next_probs, axis=0)
    
    return probs
    
    
    
def MCSprobs(probs1, probs2, probs3):
    """
    Implements a mulitple classifier system (MCS). This takes the average probability
    score over 3 independent classification algorithms' probabilities.
    
    INPUTS:
    probsX - An array of probability scores from classifier X, of size (N_samples, N_classes)
    
    OUTPUTS:
    mean_probs - An array containing the average probability scores, of size (N_samples, N_classes)
    indices - An array containing the row values of members of the testing set where at least 
                two of the classifiers disagree.
    """
    
    #Average the probability scores
    mean_probs = (probs1+probs2+probs3)/3.0
    
    #Find examples where classifiers disagree
    preds1 = np.argmax(probs1, axis=1)
    preds2 = np.argmax(probs2, axis=1)
    preds3 = np.argmax(probs3, axis=1)
    
    bool_mask = (preds1!=preds2) | (preds2!=preds3) | (preds1!=preds3)
    indices = bool_mask == True
    
    return mean_probs,  indices
    
#CURRENTLY NOT IN USE
def MCSpreds(probs1, probs2, probs3, probs4):
    preds1 = np.argmax(probs1, axis=1)
    preds2 = np.argmax(probs2, axis=1)
    preds3 = np.argmax(probs3, axis=1)
    preds4 = np.argmax(probs4, axis=1)
    
    print("preds1 uniques are: %s" %(np.unique(preds1)))
    
    combined_preds = np.concatenate((preds1, preds2, preds3, preds4), axis=1)
    
    mode_preds = mode(combined_preds, axis=1)
    
    return mode_preds
    
    
#CURRENTLY NOT IN USE
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

    
    #Run SkyNet
    
    #Read probabilities from output file
    
    
    return probs


    
    
    
    
    
    
    
    
    
