from pylab import *
from sklearn import *
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from scipy.integrate import trapz
import time
from numpy import unique

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
    
    

#SVM using the SVC routine with (currently) linear kernel (it's really slow)
#returns fpr = fp/(tp+fn) and tpr = tp/(tp+fn) and area under curve of roc
def support_vm(X_train, Y_train, X_test, Y_test):
    a=time.time()
    svr=svm.SVC(kernel='linear', probability=True)
    #svr=svm.SVC()
    #Can do a grid search of gamma as well as C
    #C_range = 10.0 ** arange(-2, 9)
    #gamma_range = 10.0 ** arange(-5, 4)
    #parameters = {'C':C_range,  'gamma':gamma_range}
    #parameters={'C':arange(1, 10)}
    #clf=grid_search.GridSearchCV(svr, parameters) #At least at the moment, this doesn't make much difference
    clf=svr
    print 'fitting now'
    f=clf.fit(X_train, Y_train)
    preds=clf.predict(X_test)
    probs= clf.predict_proba(X_test)
    #fpr, tpr,thresh = roc_curve(Y_test, probs[:, 1], pos_label=1)
    #print thresh
    #roc_auc = auc(fpr, tpr)
    #w = clf.coef_[0]
    #a = -w[0] / w[1]
    #xx = linspace(min(min(X_train[:, 0]), min(X_test[:, 0])),max(max(X_train[:, 0]), max(X_test[:, 0])))
    #yy = a * xx - (clf.intercept_[0]) / w[1]
    #plot(xx, yy, 'k')
    #show()
    
    print
    print 'Support vector machine'
    print 'Time taken', time.time()-a, 's'
    print 'Accuracy', sum(preds==Y_test)/(float)(len(preds))
    mismatched=preds[preds!=Y_test]
    
    print 'False Ia detection',  sum(mismatched==1)/(float)(sum(preds==1))
    
    #fpr2, tpr2, auc2 = roc(probs[:, 0], Y_test)
    #return fpr2, tpr2, auc2
    return probs[:, 0], Y_test
#    plot(fpr, tpr, 'b')
#    plot(fpr2, tpr2, 'g')
#    print 'Scikit AUC', roc_auc
#    print 'My AUC', auc2
#    show()
#    return fpr, tpr, roc_auc



#Can't use LinearSVC atm because it doesn't give probabilities. Stackoverflow seems to
#suggest that it would not be trivial to add my own probabilities function.
"""
#SVM using the LinearSVC routine 
#returns fpr = fp/(tp+fn) and tpr = tp/(tp+fn) and area under curve of roc
def support_vm(X_train, Y_train, X_test, Y_test):
    a=time.time()
    svr=svm.LinearSVC()
    #svr=svm.SVC()
    #Can do a grid search of gamma as well as C
    #C_range = 10.0 ** arange(-2, 9)
    #gamma_range = 10.0 ** arange(-5, 4)
    #parameters = {'C':C_range,  'gamma':gamma_range}
    #parameters={'C':arange(1, 10)}
    #clf=grid_search.GridSearchCV(svr, parameters) #At least at the moment, this doesn't make much difference
    clf=svr
    print 'fitting now'
    f=clf.fit(X_train, Y_train)
    preds=clf.predict(X_test)
    probs= clf.predict_proba(X_test)
    #fpr, tpr,thresh = roc_curve(Y_test, probs[:, 1], pos_label=1)
    #print thresh
    #roc_auc = auc(fpr, tpr)
    #w = clf.coef_[0]
    #a = -w[0] / w[1]
    #xx = linspace(min(min(X_train[:, 0]), min(X_test[:, 0])),max(max(X_train[:, 0]), max(X_test[:, 0])))
    #yy = a * xx - (clf.intercept_[0]) / w[1]
    #plot(xx, yy, 'k')
    #show()
    
    print
    print 'Support vector machine'
    print 'Time taken', time.time()-a, 's'
    print 'Accuracy', sum(preds==Y_test)/(float)(len(preds))
    mismatched=preds[preds!=Y_test]
    
    print 'False Ia detection',  sum(mismatched==1)/(float)(sum(preds==1))
    
    fpr2, tpr2, auc2 = roc(probs[:, 0], Y_test)
    return fpr2, tpr2, auc2
    
#    plot(fpr, tpr, 'b')
#    plot(fpr2, tpr2, 'g')
#    print 'Scikit AUC', roc_auc
#    print 'My AUC', auc2
#    show()
#    return fpr, tpr, roc_auc
"""


#SVM using the SVC routine with cubic kernel
#returns fpr = fp/(tp+fn) and tpr = tp/(tp+fn) and area under curve of roc
def support_vm3(X_train, Y_train, X_test, Y_test):
    a=time.time()
    svr=svm.SVC(kernel='poly', degree = 3, probability=True)
    #svr=svm.SVC()
    #Can do a grid search of gamma as well as C
    #C_range = 10.0 ** arange(-2, 9)
    #gamma_range = 10.0 ** arange(-5, 4)
    #parameters = {'C':C_range,  'gamma':gamma_range}
    #parameters={'C':arange(1, 10)}
    #clf=grid_search.GridSearchCV(svr, parameters) #At least at the moment, this doesn't make much difference
    clf=svr
    print 'fitting now'
    f=clf.fit(X_train, Y_train)
    preds=clf.predict(X_test)
    probs= clf.predict_proba(X_test)
    #fpr, tpr,thresh = roc_curve(Y_test, probs[:, 1], pos_label=1)
    #print thresh
    #roc_auc = auc(fpr, tpr)
    #w = clf.coef_[0]
    #a = -w[0] / w[1]
    #xx = linspace(min(min(X_train[:, 0]), min(X_test[:, 0])),max(max(X_train[:, 0]), max(X_test[:, 0])))
    #yy = a * xx - (clf.intercept_[0]) / w[1]
    #plot(xx, yy, 'k')
    #show()
    
    print
    print 'Support vector machine'
    print 'Time taken', time.time()-a, 's'
    print 'Accuracy', sum(preds==Y_test)/(float)(len(preds))
    mismatched=preds[preds!=Y_test]
    
    print 'False Ia detection',  sum(mismatched==1)/(float)(sum(preds==1))
    
    #fpr2, tpr2, auc2 = roc(probs[:, 0], Y_test)
    #return fpr2, tpr2, auc2
    return probs[:, 0], Y_test
#    plot(fpr, tpr, 'b')
#    plot(fpr2, tpr2, 'g')
#    print 'Scikit AUC', roc_auc
#    print 'My AUC', auc2
#    show()
#    return fpr, tpr, roc_auc

    
    
    
#A bagged RF where GridSearchCV finds the optimum number of trees between 1 and 11
#returns fpr = fp/(tp+fn) and tpr = tp/(tp+fn) and area under curve of roc
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
    probs=clf.predict_proba(X_test)[:, 0]
    #fpr, tpr, auc=roc(probs, Y_test)
    #return fpr, tpr, auc
    return probs, Y_test



#A boosted RF
#returns fpr = fp/(tp+fn) and tpr = tp/(tp+fn) and area under curve of roc
def boost_RF(X_train, Y_train, X_test, Y_test):
    a=time.time()

    classifier = AdaBoostClassifier()
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
    
    probs=classifier.predict_proba(X_test)[:, 0]
    #fpr, tpr, auc=roc(probs, Y_test)
    #return fpr, tpr, auc
    return probs, Y_test


    

#KNN classifier with weights going as 1/r
#returns fpr = fp/(tp+fn) and tpr = tp/(tp+fn) and area under curve of roc
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
    
    probs=clf.predict_proba(X_test)[:, 0]
    #fpr, tpr, auc=roc(probs, Y_test)
    #return fpr, tpr, auc
    return probs,  Y_test



#RNN classifier with weights going as 1/r
#returns fpr = fp/(tp+fn) and tpr = tp/(tp+fn) and area under curve of roc
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
        
        probs=clf.predict_proba(X_test)[:, 0]
        #fpr, tpr, auc=roc(probs, Y_test)
    except ValueError:
        print 'ValueError in RNN - probably due to no neighbours within radius'
        fpr,  tpr,  auc ,  probs,  Y_test= (None, None, None,  -9999,  None)
        
    #return fpr, tpr, auc
    return probs,  Y_test


#Naive Bayes classifier 
#returns fpr = fp/(tp+fn) and tpr = tp/(tp+fn) and area under curve of roc
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
    
    probs=clf.predict_proba(X_test)[:, 0]
    
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
    
    
    
    
    
    
    
    
