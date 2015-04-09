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





def run_ml(X_train, Y_train, X_test, Y_test, X_train_err, X_test_err, **kwargs):
    
    """
    Optimise, then run each of the classifiers, calculate ROC AUC, F1 and Kessler FoM from 
    their results. Also calculate probabilities and compare with a frequentist probability measure.
    Plots of classifier probability vs. frequency probability are produced for each classifier.
    Note the script pauses as each plot is shown and only resumes once the plot window is closed.
    
    INPUTS:
    X_train - An array of the training set features, of size (N_samples, N_features)
    Y_train - An array of the training set classes, of size (N_samples,). Class labels are 1, 2, 3
    X_test - An array of the testing set features, of size (N_samples, N_features)
    Y_test - An array of the testing set classes, of size (N_samples,). Class labels are 1, 2, 3
    X_train_err - An array of the errors on the training set features, of size (N_samples, N_features)
    X_test_err - An array of the errors on the testing set features, of size (N_samples, N_features)
    
    OUTPUTS:
    results - An array of performance criteria achieved by each classifier, of size (N_classifiers, N_criteria)
    thresholds - An array of the optimum probability threshold for each classifier wrt F1 and FoM, of size
                        (N_classifiers, 2)
    """
    
    n_features = float(X_train.shape[1])
    
    #Find optimum parameters (from a user defined dictionary of possibilities) for each classifier
    NB_params = {}
    KNN_param_dict = {'n_neighbors':[10, 15, 20, 25, 30], 'weights':['uniform', 'distance']}
    KNN_params = grid_search.KNN_optimiser(X_train, Y_train, X_test, Y_test, KNN_param_dict)
    RF_param_dict = {'n_estimators':[500, 1000, 1500], 'criterion':['gini', 'entropy']}
    RF_params = grid_search.RF_optimiser(X_train, Y_train, X_test, Y_test, RF_param_dict)
    Boost_param_dict = {'base_estimator':[RandomForestClassifier(400, 'entropy'), RandomForestClassifier(600, 'entropy')], 
                                          'n_estimators':[2, 3, 5, 10]}
    Boost_params = grid_search.Boost_optimiser(X_train, Y_train, X_test, Y_test, Boost_param_dict)
    RBF_param_dict = {'C':[0.5, 1, 2, 4], 'gamma':[1/(n_features**2), 1/n_features, 1/sqrt(n_features)]}
    RBF_params = grid_search.RBF_optimiser(X_train, Y_train, X_test, Y_test, RBF_param_dict)
    ANN_params = {}
    
    #Run classifiers with 2-fold cross validation
    probsNB=ml_algorithms.bayes(X_train,  Y_train,  X_test)
    probsNB_repeat=ml_algorithms.bayes(X_test, Y_test, X_train)
    probsKNN=ml_algorithms.nearest_neighbours(X_train, Y_train, X_test, KNN_params['n_neighbors'], KNN_params['weights'])
    probsKNN_repeat=ml_algorithms.nearest_neighbours(X_test, Y_test, X_train, KNN_params['n_neighbors'], KNN_params['weights'])
    probsRF=ml_algorithms.forest(X_train, Y_train, X_test, RF_params['n_estimators'], RF_params['criterion'])
    probsRF_repeat=ml_algorithms.forest(X_test, Y_test, X_train, RF_params['n_estimators'], RF_params['criterion'])
    probsBoost=ml_algorithms.boost_RF(X_train, Y_train, X_test, Boost_params['base_estimator'], Boost_params['n_estimators'])
    probsBoost_repeat=ml_algorithms.boost_RF(X_test, Y_test, X_train, Boost_params['base_estimator'], Boost_params['n_estimators'])
    probsRBF=ml_algorithms.support_vmRBF(X_train, Y_train, X_test, RBF_params['C'], RBF_params['gamma']) 
    probsRBF_repeat=ml_algorithms.support_vmRBF(X_test, Y_test, X_train, RBF_params['C'], RBF_params['gamma'])
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
    freq_probsNB = frequency_probabilities(X_train, Y_train, X_test, X_test_err, ml_algorithms.bayes, NB_params)
    freq_probsKNN = frequency_probabilities(X_train, Y_train, X_test, X_test_err, ml_algorithms.nearest_neighbours, 
                                            KNN_params['n_neighbors'], KNN_params['weights'])
    freq_probsRF = frequency_probabilities(X_train, Y_train, X_test, X_test_err, ml_algorithms.forest, 
                                           RF_params['n_estimators'], RF_params['criterion'])
    freq_probsBoost = frequency_probabilities(X_train, Y_train, X_test, X_test_err, 
                                              ml_algorithms.boost_RF, Boost_params['base_estimator'], Boost_params['n_estimators'])
    freq_probsRBF = frequency_probabilities(X_train, Y_train, X_test, X_test_err, 
                                            ml_algorithms.support_vmRBF, RBF_params['C'], RBF_params['gamma']) 
    #freq_probsANN = frequency_probabilities(X_train, Y_train, X_test, X_test_err, ml_algorithms.ANN, ANN_params)
    
    #Plot frequency probabilities against classifier probabilities
    plt.figure()    
    plt.scatter(freq_probsNB, probsNB[:, 0])
    plt.title('Naive Bayes - Fake Data')
    plt.xlabel('Frequency Probabilities')
    plt.ylabel('Probability Values')
    plt.savefig('/export/zupcx26/visitor4/Spring_2015/For_Wiki/prob_experiments/NB_0.7.png', facecolor='white')
    plt.show()
    
    plt.figure()
    plt.scatter(freq_probsKNN, probsKNN[:, 0])
    plt.title('K Nearest Neighbours')
    plt.xlabel('Frequency Probabilities')
    plt.ylabel('Probability Values')
    plt.savefig('/export/zupcx26/visitor4/Spring_2015/For_Wiki/prob_experiments/KNN_0.7.png', facecolor='white')
    plt.show()
    
    plt.figure()
    plt.scatter(freq_probsRF, probsRF[:, 0])
    plt.title('Random Forest')
    plt.xlabel('Frequency Probabilities')
    plt.ylabel('Probability Values')
    plt.savefig('/export/zupcx26/visitor4/Spring_2015/For_Wiki/prob_experiments/RF_0.7.png', facecolor='white')
    plt.show()
    
    plt.figure()
    plt.scatter(freq_probsBoost, probsBoost[:, 0])
    plt.title('AdaBoost Random Forest')
    plt.xlabel('Frequency Probabilities')
    plt.ylabel('Probability Values')
    plt.savefig('/export/zupcx26/visitor4/Spring_2015/For_Wiki/prob_experiments/Boost_0.7.png', facecolor='white')
    plt.show()
    
    plt.figure()
    plt.scatter(freq_probsRBF, probsRBF[:, 0])
    plt.title('Support Vector Machine with RBF Kernel')
    plt.xlabel('Frequency Probabilities')
    plt.ylabel('Probability Values')
    plt.savefig('/export/zupcx26/visitor4/Spring_2015/For_Wiki/prob_experiments/RBF_0.7.png', facecolor='white')
    plt.show()
    
    #calculate ROC curve values
    fNB, tNB, aNB=ml_algorithms.roc(probsNB, Y_test)
    fNB_repeat, tNB_repeat, aNB_repeat=ml_algorithms.roc(probsNB_repeat, Y_train)
    fKNN, tKNN, aKNN=ml_algorithms.roc(probsKNN,  Y_test)
    fKNN_repeat, tKNN_repeat, aKNN_repeat=ml_algorithms.roc(probsKNN_repeat, Y_train)
    fRF, tRF, aRF=ml_algorithms.roc(probsRF, Y_test)
    fRF_repeat, tRF_repeat, aRF_repeat=ml_algorithms.roc(probsRF_repeat, Y_train)
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
    FoMBoost_mean = 0.5*(FoMBoost+FoMBoost_repeat)
    FoMRBF_mean = 0.5*(FoMRBF+FoMRBF_repeat)
    FoMANN_mean = 0.5*(FoMANN+FoMANN_repeat)
    FoMMCS_mean = 0.5*(FoMMCS+FoMMCS_repeat)
    
    #Collate all results into a results array
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
    
    #Print best performance criteria for each classifier  
    print
    print 'AUC, F1, FoM:'
    print 'RBF SVM',  results[0, 0],  results[0, 1],  results[0, 2]
    print 'Bayes', results[1, 0],  results[1, 1], results[1, 2]
    print 'KNN', results[2, 0],  results[2, 1],  results[2, 2]
    print 'Random forest', results[3, 0],  results[3, 1],  results[3, 2]
    print 'AdaBoost forest',  results[4, 0],  results[4, 1],  results[4, 2]
    print 'ANN',  results[5, 0], results[5, 1],  results[5, 2]
    print 'MCS',  results[6, 0], results[6, 1], results[6, 2]
    print
    
    #Plot ROC curve comparing all classifiers
    plot_ROC(fRBF, tRBF, fNB, tNB, fKNN, tKNN, fRF, tRF, fBoost, tBoost, fANN, tANN, fMCS, tMCS, aRBF_mean, 
             aNB_mean, aKNN_mean, aRF_mean, aBoost_mean, aANN_mean, aMCS_mean)
    
    return results, thresholds
    
    
    
    
#Plots a ROC curve for a variety of classifier fpr and tpr vectors, and plots the mid-point of the RF vector
#i.e. the threshold =0.5 point
def plot_ROC(fRBF, tRBF, fNB, tNB, fKNN, tKNN, fRF, tRF, fBoost, tBoost, fANN, tANN, fMCS, tMCS, aRBF_mean, 
             aNB_mean, aKNN_mean, aRF_mean, aBoost_mean, aANN_mean, aMCS_mean):
    """
    Plot ROC curves comparing the performance of all the classifiers.
    
    INPUTS:
    fX - An array of the false positive rate for classifier X, of size (N_threshold_increments,)
    tX - An array of the true positive rate for classifier X, of size (N_threshold_increments,)
    aX_mean - The mean ROC AUC value for classifier X
    
    OUTPUTS:
    - A plot of the ROC curves is shown, and the script pauses until the plot window is closed
    
    """
    
    #Create figure for ROC curve
    figure(figsize=(10, 10))

    CANN='#a21d21' #brown
    CNB='#185aa9' #blue
    CKNN='#fdff00' #yellow
    CRF='#008c48' #purple
    CMCS ='#e74c3c' #red
    CBoost ='#fd85ec' #pink
    CRBF ='#40e0d0' #cyan
    
    linew=2.5

    #plot ROC curves
    plot(fRBF, tRBF, CRBF, lw=linew)
    plot(fNB, tNB, CNB, lw=linew)
    plot(fKNN, tKNN, CKNN, lw=linew)
    plot(fRF, tRF, CRF, lw=linew)
    plot(fBoost, tBoost, CBoost, lw=linew)
    plot(fANN, tANN, CANN, lw=linew)
    plot(fMCS, tMCS, CMCS, lw=linew)
    
    #plot the RF threshold = 0.5 point
    midX = int(round(fRF.shape[0]/2.0))
    midY = int(round(tRF.shape[0]/2.0))
    scatter(fRF[midX], tRF[midY], s=200, c='#000000')
    
    #Set plot parameters
    ax=gca()
    ax.set_aspect(1.0)
    setup_plot(ax)
    
    #Create legend
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
    
    
    
    
    
def frequency_probabilities(X_train, Y_train, X_test, X_test_err, classifier, *params):
    """
    A frequentist measure of the probability of each testing set member being Ia is calculated.
    This is done by taking a given testing set data point then producing many more data points
    surrounding it by perturbing its feature values (using a normal distribution with std equal to 
    the errors on each feature). This perturbed data is then classified, and the proportion 
    classified as Ia is taken as the probability of the original data point being Ia.
    
    INPUTS:
    X_train - An array containing the features of the training set, of size (N_samples, N_features)
    Y_train - An array containing the classes of the training set, of size (N_samples,)
    X_test - An array containing the features of the testing set, of size (N_samples, N_features)
    X_test_err - An array containing the errors on each feature in the testing set, of size (N_samples, N_features)
    classifier - The classification function (e.g. ml_algorithms.Bayes)
    *params - The parameters required by classifier. (e.g. the number of trees for ml_algorithms.Forest)
    
    OUTPUTS:
    freq_probs - An array containing the frequentist probabilities of each member of the testing set
                        being Ia, of size (N_samples,)
    """
    
    #Create perturbations about each data point in Y_test
    N_pert = 100
    N_classes = len(np.unique(Y_train))
    N_features = X_test.shape[1]
    perturbations = -999*np.ones((X_test.shape[0], X_test.shape[1], N_pert))
    
    for counter in np.arange(N_pert):
        perturbations[:, :, counter] = X_test+np.random.randn(X_test.shape[0], X_test.shape[1])*X_test_err

    #Classify perturbed data
    pert_probs = np.zeros((perturbations.shape[0], N_classes, perturbations.shape[2])) 
    pert_preds = np.zeros((pert_probs.shape[0], pert_probs.shape[2]))

    for counter in np.arange(perturbations.shape[0]):
        temp_data = perturbations[counter, :, :]
        temp_data = temp_data.T
        if len(params) != 2:
            temp_result = classifier(X_train, Y_train, temp_data)
        else:
            temp_result = classifier(X_train, Y_train, temp_data, params[0], params[1])
        pert_probs[counter, :, :] = temp_result.T
    
    #Convert from probability scores to class predictions
    #Note this defines 1A class as '0' regardless of whether input data is classed '1', '2', '3'
    
    #pert_preds = np.argmax(pert_probs, axis=1)
    threshold = 0.7
    for rows in arange(pert_probs.shape[0]):
        for z in arange(pert_probs.shape[2]):
            if pert_probs[rows, 0, z]>threshold:
                pert_preds[rows, z] = 0
            else:
                pert_preds[rows, z] = 1

    #Calculate probabilities of it being Ia
    freq_probs = (np.sum((pert_preds==0), axis=1)).astype(float)/pert_preds.shape[1]
    
    return freq_probs
    
    
    
def scale_data_with_errors(X, X_err):
    """
    Scale all data to be distributed about mean = 0 with std = 1. Scale the feature errors by the 
    same factor.
    
    INPUTS:
    X - An array containing all the features, of size (N_samples, N_features)
    X_err - An array containing the errors on each feature in X, of size (N_samples, N_features)
    
    OUTPUTS:
    X_scaled - An array containing all scaled features, of size (N_samples, N_features)
    X_err_scaled - An array containing all scaled errors, of size (N_samples, N_features)
    """
    
    sigma = np.std(X, axis=0)
    mean = np.mean(X, axis=0)
    X_scaled = -999*np.ones(X.shape)
    X_err_scaled = -999*np.ones(X_err.shape)
    
    for counter in np.arange(len(sigma)):
        X_scaled[:, counter] = (X[:, counter]-mean[counter]*np.ones(X.shape[0]))/sigma[counter]
        X_err_scaled[:, counter] = X_err[:, counter]/sigma[counter]
    
    return X_scaled, X_err_scaled
    
    
def make_a_fake_dataset():
    """
    Create a fake 2D data set with two distinct clouds of data corresponding
    to 2 normally distributed classes. This is for testing the rest of the classification
    pipeline.
    
    INPUTS:
    none
    
    OUPUTS:
    dataset - An array containing the fake features, of size (N_samples, 2)
    err - An array containing errors on the features in dataset, of size (N_samples, 2)
    classes - An array contianing the classes of each member of dataset, of size (N_samples,)
    """

    #Produce class distribution parameters
    means = np.array([[4., 4.], [0., 0.]])
    sigmas = np.array([[2.3, 2.1], [2.4, 1.9]])

    #Produce the feature sets
    features11 = sigmas[0, 0]*np.random.randn(100) + means[0, 0]
    features12 = sigmas[0, 1]*np.random.randn(100) + means[0, 1]
    features21 = sigmas[1, 0]*np.random.randn(100) + means[1, 0]
    features22 = sigmas[1, 1]*np.random.randn(100) + means[1, 1]
    
    weight = 0.1
    errors11 = weight*sigmas[0, 0]*np.random.randn(100)+sigmas[0, 0]
    errors12 = weight*sigmas[0, 1]*np.random.randn(100)+sigmas[0, 1]
    errors21 = weight*sigmas[1, 0]*np.random.randn(100)+sigmas[1, 0]
    errors22 = weight*sigmas[1, 1]*np.random.randn(100)+sigmas[1, 1]

    dataset = np.zeros((200, 2))
    err = np.zeros((200, 2))
    classes = np.zeros(200)

    dataset[:100, 0] = features11
    dataset[:100, 1] = features12
    dataset[100:, 0] = features21
    dataset[100:, 1] = features22
    classes[100:] = np.ones(100)
    classes = classes + 1
    
    err[:100, 0] = errors11
    err[:100, 1] = errors12
    err[100:, 0] = errors21
    err[100:, 1] = errors22
    
    #Produce the errors
    #err = np.random.randn(dataset.shape[0], dataset.shape[1])
    
    return dataset, err, classes
    
    
def misclassifications(probs, X_test, Y_test):
    """
    Make a record of the confusion matrix values for a given classifiers probability 
    scores. 
    
    INPUTS:
    probs - An array containing the probability scores for each member of the testing
                set, of size (N_samples,)
    X_test - An array containing the features of the testing set, of size (N_samples, N_features)
    Y_test - An array containing the classes of the testing set, of size (N_samples,)
    
    OUTPUTS:
    FP - The number of false positives
    FN - The number of false negatives
    TP - The number of true positives
    TN - The number of true negatives
    """
    
    #Convert from probabilities to class predictions (+1 to shift the predicted
    #class labels from 0, 1, 2 to 1, 2, 3)
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
    
    return FP, FN, TP, TN



def plot_histograms(X_test, indices1, indices2):
    """
    Plot a histogram of the distribution of chosen_feature for two sets of sample 
    indices (e.g. the lists of false positives and true positives).
    
    INPUTS:
    X_test - An array containing the features of the testing set, of size (N_samples, N_features)
    indices1 - An array of indices, of size (N_samples,)
    indices2 - An array of indices, of size (N_samples,)
    
    OUTPUTS:
    - A histogram is displayed. Note the script pauses until this plot window is closed
    """
    chosen_feature = 3
    hist1, bins1 = np.histogram(X_test[indices1, chosen_feature])
    hist1 = hist1.astype(float)/(np.amax(hist1))
    width1 = 0.7*(bins1[1]-bins1[0])
    centre1 = (bins1[:-1]+bins1[1:])/2
    my_hist1 = plt.bar(centre1, hist1, align='center', width=width1)

    show

    hist2, bins2 = np.histogram(X_test[indices2, chosen_feature])
    hist2 = hist2.astype(float)/(np.amax(hist2))
    width2 = 0.7*(bins2[1]-bins2[0])
    centre2 = (bins2[:-1]+bins2[1:])/2
    my_hist2 = plt.bar(centre2, hist2, align='center', width=width2, color='r')

    show
    
    
    
    
    
    
    
    
    
