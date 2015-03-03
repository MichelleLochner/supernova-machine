from pylab import *
from sklearn import *
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from scipy.integrate import trapz
import time
import numpy as np

#Make a roc curve to evaluate a classification routine
def roc(pr, Yt):
    probs=pr.copy()
    Y_test=Yt.copy()
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
    print("Best F1 threshold is: %s" %(best_threshold))
    
    return best_F1

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
    print("Best FoM threshold is: %s" %(best_threshold))
    
    return best_FoM



#SVM using the SVC routine with (currently) linear kernel (it's really slow)
def support_vm(X_train, Y_train, X_test, Y_test):
    
    a=time.time()
    
    svr=svm.SVC(kernel='linear', probability=True)
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

    
    return probs, Y_test



#SVM using the SVC routine with cubic kernel
def support_vm3(X_train, Y_train, X_test, Y_test):
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

    return probs, Y_test


#SVM using the SVC routine with radial basis function kernel
def support_vmRBF(X_train, Y_train, X_test, Y_test):
    a=time.time()
    svr=svm.SVC(kernel='rbf', C = 1.0, probability=True)
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

    return probs, Y_test
    
    
#A bagged RF 
def forest(X_train, Y_train, X_test, Y_test):
    a=time.time()

    #clss= RandomForestClassifier()
    #parameters=[{'n_estimators':arange(1, 20),  'max_features':arange(3, 6),  'criterion':['gini']}, 
    #                    {'n_estimators':arange(1,20), 'max_features':arange(3, 6),  'criterion':['entropy']}]
    #clf=grid_search.GridSearchCV(clss, parameters) 
    clf = RandomForestClassifier(1000, 'entropy')
    
    clf.fit(X_train, Y_train)
    #print(clf.get_params())
    preds=clf.predict(X_test)
    
    print
    print 'Random forest'
    print 'Time taken', time.time()-a, 's'
    print 'Accuracy', sum(preds==Y_test)/(float)(len(preds))
    mismatched=preds[preds!=Y_test]
    print 'False Ia detection',  sum(mismatched==1)/(float)(sum(preds==1))
    probs=clf.predict_proba(X_test)
    
    #fpr, tpr, auc=roc(probs, Y_test)
    #return fpr, tpr, auc
    return probs, Y_test



#A boosted RF
def boost_RF(X_train, Y_train, X_test, Y_test):
    a=time.time()

#    Y_train[(Y_train!=1)]=2
#    Y_test[(Y_test!=1)]=2
    
    classifier = AdaBoostClassifier(n_estimators = 400)
    #parameters={'n_estimators':arange(1, 12)}
    #clf=grid_search.GridSearchCV(clss, parameters) 
    
    classifier.fit(X_train, Y_train)
    preds=classifier.predict(X_test)
    
    print
    print 'AdaBoost forest'
    print 'Time taken', time.time()-a, 's'
    print 'Accuracy', sum(preds==Y_test)/(float)(len(preds))
    mismatched=preds[preds!=Y_test]
    print 'False Ia detection',  sum(mismatched==1)/(float)(sum(preds==1))
    
    probs=classifier.predict_proba(X_test)
    
    #print("First 10 lines of full_probs \n")
    #print(full_probs[:9, :])
    
    #print("First 10 lines of preds \n")
    #print(preds[:9])
    
    #rounded_probs = np.around(probs)
    #compare_list = preds-probs
    #compare = np.sum(compare_list)
    
    #unique_probs = np.unique(probs)
    #unique_preds = np.unique(preds)
    #print("Unique prediction values are: %s" %(unique_preds))
    #print("Unique prob values are: %s" %(unique_probs))
    #print("total difference between preds and probs: %s" %(compare))
    
    return probs, Y_test


    

#KNN classifier with weights going as 1/r
def nearest_neighbours(X_train, Y_train, X_test, Y_test):
    a=time.time()
    print 'Size of training set is',  len(X_train[:, 0])
    n_neighbors=20
    clf=neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(X_train, Y_train)
    preds=clf.predict(X_test)
    
    print
    print 'K nearest neighbours'
    print 'Time taken', time.time()-a, 's'
    print 'Accuracy', sum(preds==Y_test)/(float)(len(preds))
    mismatched=preds[preds!=Y_test]
    print 'False Ia detection',  sum(mismatched==1)/(float)(sum(preds==1))
    
    probs=clf.predict_proba(X_test)
    #fpr, tpr, auc=roc(probs, Y_test)
    #return fpr, tpr, auc
    return probs,  Y_test



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
    return probs,  Y_test


#Naive Bayes classifier 
def bayes(X_train, Y_train, X_test, Y_test):
    a=time.time()
    clf = GaussianNB()
    f=clf.fit(X_train, Y_train)
    preds=array(clf.predict(X_test), dtype='int')
    
    print
    print 'Naive Bayes'
    print 'Time taken', time.time()-a, 's'
    print 'Accuracy', sum(preds==Y_test)/(float)(len(preds))
    mismatched=preds[preds!=Y_test]
    print 'False Ia detection',  sum(mismatched==1)/(float)(sum(preds==1))
    
    probs=clf.predict_proba(X_test)
    
    #fpr, tpr, auc=roc(probs, Y_test)
    #return fpr, tpr, auc
    return probs,  Y_test
    
    
    
    
#    y_score=f.decision_function(X_test)[:, 0]
#    fpr, tpr, _ = roc_curve(Y_test, y_score, pos_label=1)
#    roc_auc = auc(fpr, tpr)
#    return fpr, tpr, roc_auc
#    P=clf.predict_proba(X_test)
#    
#    inds=where((preds==Y_test)&(preds==1))[0]
#    figure()
#    hist(P[inds, 0], 50)
#    xlabel('1')
#    
#    inds=where((preds==Y_test)&(preds==2))[0]
#    figure()
#    hist(P[inds, 1], 50)
#    xlabel('2')
#    
#    inds=where((preds==Y_test)&(preds==3))[0]
#    figure()
#    hist(P[inds, 2], 50)
#    xlabel('3')
#    show()
    
    
    
    
    
    
    
    
