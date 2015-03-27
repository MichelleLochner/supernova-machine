from pylab import *
from sklearn import *
import max_ml_algorithms as ml_algorithms
import pywt, os, math, time
from sklearn.decomposition import PCA, KernelPCA,  SparsePCA,  FastICA
from sklearn.lda import LDA
import numpy as np
import max_grid_search as grid_search
from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
#from gapp import dgp

rcParams['font.family']='serif'

filters=['g','r','i','z']
#colours={'u':'#6614de','g':'#007718','r':'#b30100','i':'#d35c00','z':'#d2003d','Y':'#000000'}
colours={'u':'#6614de','g':'#007718','r':'#b30100','i':'#d35c00','z':'k','Y':'#000000'}
MAX_LEN=175+2 #The longest observation we have


#Returns the light curve in each band as a dictionary object, given a full file path
def get_lightcurve(flname):
    global filters
    fl=open(flname,'r')
    mjd=[]
    flt=[]
    flux=[]
    fluxerr=[]
    z=-9
    z_err=-9
    for line in fl:
        s=line.split()
        if len(s)>0:
            if s[0]=='HOST_GALAXY_PHOTO-Z:':
                z=(float)(s[1])
                z_err=(float)(s[3])
            elif s[0]=='OBS:':
                mjd.append((float)(s[1]))
                flt.append(s[2])
                flux.append((float)(s[4]))
                fluxerr.append((float)(s[5]))
            elif s[0]=='SNTYPE:':
                type=(int)(s[1])
    
    flt=array(flt)            
    mjd=array(mjd)
    flux=array(flux)
    fluxerr=array(fluxerr)
    d={}
    for i in range(len(filters)):
        inds=where(flt==filters[i])
#        if i==0:
#            print mjd[inds]
        X=column_stack((mjd[inds],flux[inds],fluxerr[inds]))
        d[filters[i]]=X
        
    return d,z, z_err, type
   
  
#Fits a cubic spine between only two points given x and y input and their derivatives and evaluates it on x_eval
def fit_spline(x, y, d, x_eval):
    if len(x)>2:
        print 'this is only appropriate for 2 datapoints'
        return 0
        
    x1, x2=x
    y1, y2=y
    k1, k2=d
    
    a=k1*(x2-x1)-(y2-y1)
    b=-k2*(x2-x1)+(y2-y1)
    
    t=(x_eval-x1)/(x2-x1)
    return (1-t)*y1+t*y2+t*(1-t)*(a*(1-t)+b*t)

#Gets the features for our spectroscopically confirmed set
def spectro_set():
    root='Simulations/SIMGEN_PUBLIC_DES/'
    fls=os.listdir(root)
    fls.sort()
    spec_set=[]
    for i in range(len(fls)):
        if fls[i][-3:]=='DAT':
            f=open(root+fls[i], 'r')
            for line in f:
                s=line.split()
                if len(s)>0 and s[0]=='SNTYPE:':
                    if (int)(s[1])!=-9:
                        spec_set.append(fls[i])
            f.close()
    print "Number of SNe", len(spec_set)
    savetxt('spectro.list', spec_set, fmt='%s')
    
def iswt(coefficients, wavelet):
    """
      M. G. Marino to complement pyWavelets' swt.
      Input parameters:

        coefficients
          approx and detail coefficients, arranged in level value
          exactly as output from swt:
          e.g. [(cA1, cD1), (cA2, cD2), ..., (cAn, cDn)]

        wavelet
          Either the name of a wavelet or a Wavelet object

    """
    output = coefficients[0][0].copy() # Avoid modification of input data

    #num_levels, equivalent to the decomposition level, n
    num_levels = len(coefficients)
    for j in range(num_levels,0,-1):
        step_size = int(math.pow(2, j-1))
        last_index = step_size
        _, cD = coefficients[num_levels - j]
        for first in range(last_index): # 0 to last_index - 1

            # Getting the indices that we will transform
            indices = arange(first, len(cD), step_size)

            # select the even indices
            even_indices = indices[0::2]
            # select the odd indices
            odd_indices = indices[1::2]

            # perform the inverse dwt on the selected indices,
            # making sure to use periodic boundary conditions
            x1 = pywt.idwt(output[even_indices], cD[even_indices], wavelet, 'per')
            x2 = pywt.idwt(output[odd_indices], cD[odd_indices], wavelet, 'per')

            # perform a circular shift right
            x2 = roll(x2, 1)

            # average and insert into the correct indices
            output[indices] = (x1 + x2)/2.  

    return output
  
#Given a matrix of features (so each row is an observation, column a feature), returns the eigenvectors
#and eigenvalues
def pca(X):
    #Find the normalised spectra
    X=X.transpose()
    x_norm=X.copy()
    for c in range(len(X[0, :])):
        nor=sqrt(sum(X[:, c]**2))
        if nor==0:
            print 'column norm=0', c
            x_norm[:, c]=0
        else:
            x_norm[:, c]=X[:, c]/nor
        if isnan(sum(x_norm[:, c])):
            print 'column', c
            print X[:, c]
            sys.exit(0)
    #Construct the covariance matrix
    C=zeros([len(X[:, 0]), len(X[:, 0])])
    for i in range(len(C[:, 0])):
        for j in range(len(C[0, :])):
            C[i, j]=sum(x_norm[i, :]*x_norm[j, :])
    
    #Use Singular Value Decomposition to get the eigenvectors and eigenvalues
    C=mat(C)
    vals, vec = linalg.eigh(C)
    inds=argsort(vals)[::-1]
    return vals[inds], vec[:, inds]
    
#Given a feature vector and matrix of eigenvectors (in columns), returns the coefficients
def pca_coeffs(X, eigs):
   # print shape(mat(eigs)), shape(mat(X).T)
    t1=time.time()
#    try:
    A=linalg.lstsq(mat(eigs), mat(X).T)[0].flatten()
#    except ValueError:
#        print eigs
#    print 'time for ls', time.time()-t1
#    print shape(A)
    return A
    
#Figure out how many components we need
def best_coeffs(vals):
    tot=sum(vals)
    tol=0.98
    tot2=0
    for i in range(len(vals)):
        tot2+=vals[i]
        if tot2>=tol*tot:
            return i
            break
            
    print "something's wrong with the PCA"
    return -1

#Plots a light curve
def plot_lc(fname):
    d, z, zerr, type=get_lightcurve(fname)
    
    for j in range(len(filters)):
        #subplot(2, 2, j+1)
        X=d[filters[j]]
        errorbar(X[:, 0]-min(X[:, 0]), X[:, 1],yerr=X[:, 2],  marker='o',linestyle='none',  color=colours[filters[j]])
    show()

#Some setup parameters to make the plot pretty
def setup_plot(ax):
    
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    #locs, lbs = xticks()
    #setp(lbs, rotation=45)

#    ax.spines["right"].set_visible(False)
#    ax.spines["top"].set_visible(False)
#    ax.xaxis.set_ticks_position('bottom')
#    ax.yaxis.set_ticks_position('left')
    
    fontszy=14
    fontszx=14
    for tick in ax.xaxis.get_major_ticks():
           tick.label1.set_fontsize(fontszx)
    for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontszy)
    #subplots_adjust(bottom=0.25,left=0.15)


#Find the misclassified examples
def misclassifications(probs, X_test, Y_test):
    
    preds = argmax(probs, axis=1) + 1
    
    FP_indices = (preds == 1) & (Y_test != 1)
    FN_indices = (preds != 1) & (Y_test == 1)
    TP_indices = (preds == 1) & (Y_test == 1)
    TN_indices = (preds != 1) & (Y_test != 1)
    
    #plot_histograms(X_test, FP_indices, TP_indices)
    
    FP = X_test[FP_indices, :]
    FN = X_test[FN_indices, :]
    TP = X_test[TP_indices, :]
    TN = X_test[TN_indices, :]
    
    #print('The number of RF FPs: %s' %(RF_FP.shape[0]))
    #print('The number of RF FNs: %s' %(RF_FN.shape[0]))
    #print('The number of RF TPs: %s' %(RF_TP.shape[0]))
    #print('The number of RF TNs: %s' %(RF_TN.shape[0]))
    
    return FP, FN, TP, TN



def plot_histograms(X_test, indices1, indices2):
    
    hist1, bins1 = np.histogram(X_test[indices1, 3])
    hist1 = hist1.astype(float)/(np.amax(hist1))
    width1 = 0.7*(bins1[1]-bins1[0])
    centre1 = (bins1[:-1]+bins1[1:])/2
    my_hist1 = plt.bar(centre1, hist1, align='center', width=width1)

    show

    hist2, bins2 = np.histogram(X_test[indices2, 3])
    hist2 = hist2.astype(float)/(np.amax(hist2))
    width2 = 0.7*(bins2[1]-bins2[0])
    centre2 = (bins2[:-1]+bins2[1:])/2
    my_hist2 = plt.bar(centre2, hist2, align='center', width=width2, color='r')

    show



#Run classification algorithms, with or without the SVM (it's slow)
def run_ml(X_train, Y_train, X_test, Y_test, X_train_err, X_test_err, **kwargs):
    
    n_features = float(X_train.shape[1])
    
    #Lets try optimising KNN
    KNN_param_dict = {'n_neighbors':[10, 15, 20, 25, 30], 'weights':['uniform', 'distance']}
    KNN_params = grid_search.KNN_optimiser(X_train, Y_train, X_test, Y_test, KNN_param_dict)
    RF_param_dict = {'n_estimators':[500, 1000, 1500], 'criterion':['gini', 'entropy']}
    RF_params = grid_search.RF_optimiser(X_train, Y_train, X_test, Y_test, RF_param_dict)
    Boost_param_dict = {'base_estimator':[RandomForestClassifier(400, 'entropy'), RandomForestClassifier(600, 'entropy')], 
                                          'n_estimators':[2, 3, 5, 10]}
    Boost_params = grid_search.Boost_optimiser(X_train, Y_train, X_test, Y_test, Boost_param_dict)
    RBF_param_dict = {'C':[0.5, 1, 2, 4], 'gamma':[1/(n_features**2), 1/n_features, 1/sqrt(n_features)]}
    RBF_params = grid_search.RBF_optimiser(X_train, Y_train, X_test, Y_test, RBF_param_dict)
    
    #Run classifiers with 2-fold cross validation
    probsNB=ml_algorithms.bayes(X_train,  Y_train,  X_test,  Y_test)
    probsNB_repeat=ml_algorithms.bayes(X_test, Y_test, X_train, Y_train)
    probsKNN=ml_algorithms.nearest_neighbours(X_train, Y_train, X_test, Y_test, KNN_params['n_neighbors'], KNN_params['weights'])
    probsKNN_repeat=ml_algorithms.nearest_neighbours(X_test, Y_test, X_train, Y_train, KNN_params['n_neighbors'], KNN_params['weights'])
    probsRF=ml_algorithms.forest(X_train, Y_train, X_test, Y_test, RF_params['n_estimators'], RF_params['criterion'])
    probsRF_repeat=ml_algorithms.forest(X_test, Y_test, X_train, Y_train, RF_params['n_estimators'], RF_params['criterion'])
    #probsRNN=ml_algorithms.radius_neighbours(X_train, Y_train, X_test, Y_test)
    #probsRNN_repeat=ml_algorithms.radius_neighbours(X_test, Y_test, X_train, Y_train)
    probsBoost=ml_algorithms.boost_RF(X_train, Y_train, X_test, Y_test, Boost_params['base_estimator'], Boost_params['n_estimators'])
    probsBoost_repeat=ml_algorithms.boost_RF(X_test, Y_test, X_train, Y_train, Boost_params['base_estimator'], Boost_params['n_estimators'])
    probsRBF=ml_algorithms.support_vmRBF(X_train, Y_train, X_test, Y_test, RBF_params['C'], RBF_params['gamma'])
    probsRBF_repeat=ml_algorithms.support_vmRBF(X_test, Y_test, X_train, Y_train, RBF_params['C'], RBF_params['gamma'])
    probsANN=ml_algorithms.ANN(X_train, Y_train, X_test, Y_test)
    probsANN_repeat=ml_algorithms.ANN(X_test, Y_test, X_train, Y_train)
    probs_MCS,  hard_indices_test = ml_algorithms.MCSprobs(probsRF, probsRBF,probsBoost)
    probs_MCS_repeat,  hard_indices_train = ml_algorithms.MCSprobs(probsRF_repeat, probsRBF_repeat, probsBoost_repeat)

    #make a record of examples the classifiers in MCS disagreed on
    hard_X_train = X_train[hard_indices_train, :]
    hard_Y_train = Y_train[hard_indices_train]
    hard_X_test = X_test[hard_indices_test, :]
    hard_Y_test = Y_test[hard_indices_test]
    
    #make a record of the FP and FN from RF
    RF_FP, RF_FN,  RF_TP, RF_TN = misclassifications(probsRF, X_test, Y_test)
    
    #Calculate frequency probabilities
    freq_probsNB = frequency_probabilities(X_train, Y_train, X_test, Y_test, X_test_err, None)

    print(freq_probsNB.shape)
    print(freq_probsNB[:30])
    print(probsNB[:30])
    
    plt.scatter(freq_probsNB, probsNB[:, 0])
    plt.xlabel('Frequency Probabilities')
    plt.ylabel('Probability Values')
    plt.show()

    #calculate ROC curve values
    fNB, tNB, aNB=ml_algorithms.roc(probsNB, Y_test)
    fNB_repeat, tNB_repeat, aNB_repeat=ml_algorithms.roc(probsNB_repeat, Y_train)
    fKNN, tKNN, aKNN=ml_algorithms.roc(probsKNN,  Y_test)
    fKNN_repeat, tKNN_repeat, aKNN_repeat=ml_algorithms.roc(probsKNN_repeat, Y_train)
    fRF, tRF, aRF=ml_algorithms.roc(probsRF, Y_test)
    fRF_repeat, tRF_repeat, aRF_repeat=ml_algorithms.roc(probsRF_repeat, Y_train)
    #fRNN, tRNN, aRNN=ml_algorithms.roc(probsRNN, Y_test)
    #fRNN_repeat, tRNN_repeat, aRNN_repeat=ml_algorithms.roc(probsRNN_repeat, Y_train)
    fBoost, tBoost, aBoost=ml_algorithms.roc(probsBoost, Y_test)
    fBoost_repeat, tBoost_repeat, aBoost_repeat=ml_algorithms.roc(probsBoost_repeat, Y_train)
    fRBF, tRBF, aRBF=ml_algorithms.roc(probsRBF, Y_test)
    fRBF_repeat, tRBF_repeat, aRBF_repeat=ml_algorithms.roc(probsRBF_repeat, Y_train)
    fANN, tANN, aANN=ml_algorithms.roc(probsANN, Y_test)
    fANN_repeat, tANN_repeat, aANN_repeat=ml_algorithms.roc(probsANN_repeat, Y_train)
    fMCS,  tMCS, aMCS = ml_algorithms.roc(probs_MCS, Y_test)
    fMCS_repeat, tMCS_repeat, aMCS_repeat = ml_algorithms.roc(probs_MCS_repeat, Y_train)

    #calculate mean AUC over cross validation
    aNB_mean = (aNB+aNB_repeat)/2.0
    aKNN_mean = (aKNN+aKNN_repeat)/2.0
    aRF_mean = (aRF+aRF_repeat)/2.0
    #aRNN_mean = (aRNN+aRNN_repeat)/2.0
    aBoost_mean = (aBoost+aBoost_repeat)/2.0
    aRBF_mean = (aRBF+aRBF_repeat)/2.0
    aANN_mean = (aANN+aANN_repeat)/2.0
    aMCS_mean = (aMCS+aMCS_repeat)/2.0
    
    #calculate F1 values
    F1_scoreNB, F1_thresholdNB = ml_algorithms.F1(probsNB, Y_test)
    F1_scoreNB_repeat, F1_thresholdNB_repeat = ml_algorithms.F1(probsNB_repeat, Y_train)
    F1_scoreKNN, F1_thresholdKNN = ml_algorithms.F1(probsKNN, Y_test)
    F1_scoreKNN_repeat, F1_thresholdKNN_repeat = ml_algorithms.F1(probsKNN_repeat, Y_train)
    F1_scoreRF,  F1_thresholdRF = ml_algorithms.F1(probsRF, Y_test)
    F1_scoreRF_repeat,  F1_thresholdRF_repeat = ml_algorithms.F1(probsRF_repeat, Y_train)
    #F1_scoreRNN = ml_algorithms.F1(probsRNN, Y_test)
    #F1_scoreRNN_repeat = ml_algorithms.F1(probsRNN_repeat, Y_train)
    F1_scoreBoost,  F1_thresholdBoost = ml_algorithms.F1(probsBoost, Y_test)
    F1_scoreBoost_repeat,  F1_thresholdBoost_repeat = ml_algorithms.F1(probsBoost_repeat, Y_train)
    F1_scoreRBF, F1_thresholdRBF = ml_algorithms.F1(probsRBF, Y_test)
    F1_scoreRBF_repeat, F1_thresholdRBF_repeat = ml_algorithms.F1(probsRBF_repeat, Y_train)
    F1_scoreANN, F1_thresholdANN = ml_algorithms.F1(probsANN, Y_test)
    F1_scoreANN_repeat, F1_thresholdANN_repeat = ml_algorithms.F1(probsANN_repeat, Y_train)
    F1_scoreMCS, F1_thresholdMCS = ml_algorithms.F1(probs_MCS, Y_test)
    F1_scoreMCS_repeat, F1_thresholdMCS_repeat = ml_algorithms.F1(probs_MCS_repeat, Y_train)
    
    F1_scoreNB_mean = 0.5*(F1_scoreNB+F1_scoreNB_repeat)
    F1_scoreKNN_mean = 0.5*(F1_scoreKNN+F1_scoreKNN_repeat)
    F1_scoreRF_mean = 0.5*(F1_scoreRF+F1_scoreRF_repeat)
    #F1_scoreRNN_mean = 0.5*(F1_scoreRNN+F1_scoreRNN_repeat)
    F1_scoreBoost_mean = 0.5*(F1_scoreBoost+F1_scoreBoost_repeat)
    F1_scoreRBF_mean = 0.5*(F1_scoreRBF+F1_scoreRBF_repeat)
    F1_scoreANN_mean = 0.5*(F1_scoreANN+F1_scoreANN_repeat)
    F1_scoreMCS_mean = 0.5*(F1_scoreMCS+F1_scoreMCS_repeat)
    
    #calculate Kessler FoM values
    FoMNB, FoM_thresholdNB = ml_algorithms.FoM(probsNB, Y_test)
    FoMNB_repeat, FoM_thresholdNB_repeat = ml_algorithms.FoM(probsNB_repeat, Y_train)
    FoMKNN, FoM_thresholdKNN = ml_algorithms.FoM(probsKNN, Y_test)
    FoMKNN_repeat, FoM_thresholdKNN_repeat = ml_algorithms.FoM(probsKNN_repeat, Y_train)
    FoMRF, FoM_thresholdRF = ml_algorithms.FoM(probsRF, Y_test)
    FoMRF_repeat, FoM_thresholdRF_repeat = ml_algorithms.FoM(probsRF_repeat, Y_train)
    #FoMRNN = ml_algorithms.FoM(probsRNN,Y_test)
    #FoMRNN_repeat = ml_algorithms.FoM(probsRNN_repeat,Y_train)
    FoMBoost, FoM_thresholdBoost = ml_algorithms.FoM(probsBoost, Y_test)
    FoMBoost_repeat, FoM_thresholdBoost_repeat = ml_algorithms.FoM(probsBoost_repeat, Y_train)
    FoMRBF, FoM_thresholdRBF = ml_algorithms.FoM(probsRBF, Y_test)
    FoMRBF_repeat, FoM_thresholdRBF_repeat = ml_algorithms.FoM(probsRBF_repeat, Y_train)
    FoMANN, FoM_thresholdANN = ml_algorithms.FoM(probsANN, Y_test)
    FoMANN_repeat, FoM_thresholdANN_repeat = ml_algorithms.FoM(probsANN_repeat, Y_train)
    FoMMCS, FoM_thresholdMCS = ml_algorithms.FoM(probs_MCS, Y_test)
    FoMMCS_repeat, FoM_thresholdMCS_repeat = ml_algorithms.FoM(probs_MCS_repeat, Y_train)
    
    FoMNB_mean = 0.5*(FoMNB+FoMNB_repeat)
    FoMKNN_mean = 0.5*(FoMKNN+FoMKNN_repeat)
    FoMRF_mean = 0.5*(FoMRF+FoMRF_repeat)
    #FoMRNN_mean = 0.5*(FoMRNN+FoMRNN_repeat)
    FoMBoost_mean = 0.5*(FoMBoost+FoMBoost_repeat)
    FoMRBF_mean = 0.5*(FoMRBF+FoMRBF_repeat)
    FoMANN_mean = 0.5*(FoMANN+FoMANN_repeat)
    FoMMCS_mean = 0.5*(FoMMCS+FoMMCS_repeat)
    
    #alt_FoMRF = ml_algorithms.alternative_FoM(probsRF, Y_test)
    #print('Alternative RF FoM is: %s' %(alt_FoMRF))
    
    #Collate all results into a results array. Columns are AUC, F1, FoM, rows are classifiers
    results = np.array([[aRBF_mean, F1_scoreRBF_mean, FoMRBF_mean], 
                                   [aNB_mean, F1_scoreNB_mean, FoMNB_mean], 
                                   [aKNN_mean, F1_scoreKNN_mean, FoMKNN_mean], 
                                   [aRF_mean, F1_scoreRF_mean, FoMRF_mean], 
                                   [aBoost_mean, F1_scoreBoost_mean, FoMBoost_mean], 
                                   [aANN_mean, F1_scoreANN_mean, FoMANN_mean], 
                                   [aMCS_mean, F1_scoreMCS_mean, FoMMCS_mean]])

    thresholds = np.array([[F1_thresholdRBF, FoM_thresholdRBF], 
                                        [F1_thresholdNB, FoM_thresholdNB], 
                                        [F1_thresholdKNN, FoM_thresholdKNN], 
                                        [F1_thresholdRF, FoM_thresholdRF], 
                                        [F1_thresholdBoost, FoM_thresholdBoost], 
                                        [F1_thresholdANN, FoM_thresholdANN], 
                                        [F1_thresholdMCS, FoM_thresholdMCS]])
    
    """
    print
    print 'AUC, F1, FoM:'
    print 'RBF SVM',  results[0, 0],  results[0, 1],  results[0, 2]
    print 'Bayes', results[1, 0],  results[1, 1], results[1, 2]
    print 'KNN', results[2, 0],  results[2, 1],  results[2, 2]
    print 'Random forest', results[3, 0],  results[3, 1],  results[3, 2]
    #print 'RNN',  aRNN, F1_scoreRNN_mean, FoMRNN_mean
    print 'AdaBoost forest',  results[4, 0],  results[4, 1],  results[4, 2]
    print 'ANN',  results[5, 0], results[5, 1],  results[5, 2]
    print 'MCS',  results[6, 0], results[6, 1], results[6, 2]
    print
    """
    
    #Plot ROC curve
    #plot_ROC(fRBF, tRBF, fNB, tNB, fKNN, tKNN, fRF, tRF, fBoost, tBoost, fANN, tANN, fMCS, tMCS, aRBF_mean, 
    #         aNB_mean, aKNN_mean, aRF_mean, aBoost_mean, aANN_mean, aMCS_mean)
    
    
    return results, thresholds
    
    
    
    
#Plots a ROC curve for a variety of classifier fpr and tpr vectors, and plots the mid-point of the RF vector
#i.e. the threshold =0.5 point
def plot_ROC(fRBF, tRBF, fNB, tNB, fKNN, tKNN, fRF, tRF, fBoost, tBoost, fANN, tANN, fMCS, tMCS, aRBF_mean, 
             aNB_mean, aKNN_mean, aRF_mean, aBoost_mean, aANN_mean, aMCS_mean):
    #Create figure for ROC curve
    figure(figsize=(10, 10))

    CANN='#a21d21' #brown
    CNB='#185aa9' #blue
    CKNN='#fdff00' #yellow
    CRF='#008c48' #purple
    CMCS ='#e74c3c' #red
    CBoost ='#fd85ec' #pink
    #CRNN ='#a27e2c' #brown
    CRBF ='#40e0d0' #cyan
    
    linew=2.5

    #plot ROC curves
    plot(fRBF, tRBF, CRBF, lw=linew)
    plot(fNB, tNB, CNB, lw=linew)
    plot(fKNN, tKNN, CKNN, lw=linew)
    plot(fRF, tRF, CRF, lw=linew)
    #plot(fRNN, tRNN, CRNN, lw=linew)
    plot(fBoost, tBoost, CBoost, lw=linew)
    plot(fANN, tANN, CANN, lw=linew)
    plot(fMCS, tMCS, CMCS, lw=linew)
    
    #plot the threshold = 0.5 point
    midX = int(round(fRF.shape[0]/2.0))
    midY = int(round(tRF.shape[0]/2.0))
    scatter(fRF[midX], tRF[midY], s=200, c='#000000')

    
    #Set plot parameters
    ax=gca()
    ax.set_aspect(1.0)
    setup_plot(ax)
    
    legend(('RBF SVM (%.3f)' %(aRBF_mean),  'Naive Bayes (%.3f)' %(aNB_mean), 'KNN (%.3f)' %(aKNN_mean), 
    'Random Forest (%.3f)' %(aRF_mean), 'Ada Forest (%.3f)' %(aBoost_mean), 'ANN (%.3f)' %(aANN_mean), 
    'MCS (%.3f)' %(aMCS_mean)),  loc='lower right',  frameon=True, bbox_to_anchor=(0.95, 0.05), fontsize=18)
    
    title('ROC Curve', fontsize=22)
    xlabel('False positive rate (contamination)', fontsize=18)
    ylabel('True positive rate (recall)', fontsize=18)
    xlim([0, 1])
    ylim([0, 1])
    
    subplots_adjust(bottom=0.08,left=0.05, top=0.92, right=0.95)
    show()
    

#Introduce bias into training set - doesn't work currently
def split_data(X, Y, size, bias):
    
    if bias != None:
        Pos_indices = Y==1
        Neg_indices = Y!=1
        
        Positives = X[Pos_indices, :]
        Negatives = X[Neg_indices, :]
        Pos_classes = np.expand_dims(Y[Pos_indices], axis=1)
        Neg_classes = np.expand_dims(Y[Neg_indices], axis=1)
        
        Positives = np.append(Positives, Pos_classes, axis=1)
        Negatives = np.append(Negatives, Neg_classes, axis=1)
        
        Neg_proportion = (0.5*X.shape[0]-bias*Positives.shape[0])/Negatives.shape[0]
        
        Pos_train, Pos_test = train_test_split(Positives, test_size=1.0-bias, random_state = np.random.randint(100))
        Neg_train, Neg_test = train_test_split(Negatives, test_size = 1.0-Neg_proportion, random_state = np.random.randint(100))
        
        X_train = np.append(Pos_train[:, :-1], Neg_train[:, :-1], axis=0)
        Y_train = np.append(Pos_train[:, -1], Neg_train[:, -1], axis=0)
        X_test = np.append(Pos_test[:, :-1], Neg_train[:, :-1], axis=0)
        Y_test = np.append(Pos_test[:, -1], Neg_train[:, -1], axis=0)
        
        train_index = np.arange(X_train.shape[0])
        np.random.shuffle(train_index)
        test_index = np.arange(X_test.shape[0])
        np.random.shuffle(test_index)
        
        X_train = X_train[train_index, :]
        Y_train = Y_train[train_index]
        X_test = X_test[test_index, :]
        Y_test = Y_test[test_index]
        
        
        
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=size, random_state=np.random.randint(100))
    
    return X_train, Y_train, X_test, Y_test
    
    
    
    
    
def frequency_probabilities(X_train, Y_train, X_test, Y_test, X_test_err, params):
        
    #Create perturbations about each data point in Y_test
    N_pert = 1000
    N_classes = len(np.unique(Y_test))
    N_features = X_test.shape[1]

    perturbations = -999*np.ones((X_test.shape[0], X_test.shape[1], N_pert))
    
    for counter in np.arange(N_pert):
        perturbations[:, :, counter] = X_test+np.random.randn(X_test.shape[0], X_test.shape[1])*X_test_err

    #Classify perturbed data
    pert_probs = np.zeros((perturbations.shape[0], N_classes, perturbations.shape[2])) 

    for counter in np.arange(perturbations.shape[0]):
        temp_data = perturbations[counter, :, :]
        temp_data = temp_data.T
        temp_result = ml_algorithms.support_vm(X_train, Y_train, temp_data, None)
        pert_probs[counter, :, :] = temp_result.T
    

    pert_preds = np.argmax(pert_probs, axis=1)

    #Calculate probabilities of it being 1A
    freq_probs = (np.sum((pert_preds==1), axis=1)).astype(float)/pert_preds.shape[1]
    
    return freq_probs
    
    
    
def scale_data_with_errors(X, X_err):
        
    sigma = np.std(X, axis=0)
    mean = np.mean(X, axis=0)
    X_scaled = -999*np.ones(X.shape)
    X_err_scaled = -999*np.ones(X_err.shape)
    
    for counter in np.arange(len(sigma)):
        X_scaled[:, counter] = (X[:, counter]-mean[counter]*np.ones(X.shape[0]))/sigma[counter]
        X_err_scaled[:, counter] = X_err[:, counter]/sigma[counter]
    
    return X_scaled, X_err_scaled
    
    
    
    
    
    
    
    
