from pylab import *
from sklearn import *
import max_ml_algorithms as ml_algorithms
import pywt, os, math, time
from sklearn.decomposition import PCA, KernelPCA,  SparsePCA,  FastICA
from sklearn.lda import LDA
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



#Run classification algorithms, with or without the SVM (it's slow)
def run_ml(X_train, Y_train, X_test, Y_test, **kwargs):
    SVM=False
    if 'run_svm' in kwargs and kwargs['run_svm']==True:
       SVM=True 
    
    #Run classifiers with 2-fold cross validation
    probs2, Y_test2=ml_algorithms.bayes(X_train,  Y_train,  X_test,  Y_test)
    probs2_repeat, Y_test2_repeat=ml_algorithms.bayes(X_test, Y_test, X_train, Y_train)
    probs3, Y_test3=ml_algorithms.nearest_neighbours(X_train, Y_train, X_test, Y_test)
    probs3_repeat, Y_test3_repeat=ml_algorithms.nearest_neighbours(X_test, Y_test, X_train, Y_train)
    probs4, Y_test4=ml_algorithms.forest(X_train, Y_train, X_test, Y_test)
    probs4_repeat, Y_test4_repeat=ml_algorithms.forest(X_test, Y_test, X_train, Y_train)
    probs5, Y_test5=ml_algorithms.radius_neighbours(X_train, Y_train, X_test, Y_test)
    probs5_repeat, Y_test5_repeat=ml_algorithms.radius_neighbours(X_test, Y_test, X_train, Y_train)
    probs6, Y_test6=ml_algorithms.boost_RF(X_train, Y_train, X_test, Y_test)
    probs6_repeat, Y_test6_repeat=ml_algorithms.boost_RF(X_test, Y_test, X_train, Y_train)

    if SVM:
        probs1, Y_test1=ml_algorithms.support_vm(X_train, Y_train, X_test, Y_test)
        probs1_repeat, Y_test1_repeat=ml_algorithms.support_vm(X_test, Y_test, X_train, Y_train)
        probs7, Y_test7=ml_algorithms.support_vm3(X_train, Y_train, X_test, Y_test)
        probs7_repeat, Y_test7_repeat=ml_algorithms.support_vm3(X_test, Y_test, X_train, Y_train)
        probs8, Y_test8=ml_algorithms.support_vmRBF(X_train, Y_train, X_test, Y_test)
        probs8_repeat, Y_test8_repeat=ml_algorithms.support_vmRBF(X_test, Y_test, X_train, Y_train)
        
        #calculate ROC curve values
        f1, t1, a1=ml_algorithms.roc(probs1, Y_test1)
        f1_repeat, t1_repeat, a1_repeat=ml_algorithms.roc(probs1_repeat, Y_test1_repeat)
        f7, t7, a7=ml_algorithms.roc(probs7, Y_test7)
        f7_repeat, t7_repeat, a7_repeat=ml_algorithms.roc(probs7_repeat, Y_test7_repeat)
        f8, t8, a8=ml_algorithms.roc(probs8, Y_test8)
        f8_repeat, t8_repeat, a8_repeat=ml_algorithms.roc(probs8_repeat, Y_test8_repeat)
        
        a1_mean = (a1+a1_repeat)/2.0
        a7_mean = (a7+a7_repeat)/2.0
        a8_mean = (a8+a8_repeat)/2.0
        
        #calculate F1 values
        F1_score1 = ml_algorithms.F1(probs1, Y_test1)
        F1_score1_repeat = ml_algorithms.F1(probs1_repeat, Y_test1_repeat)
        F1_score7 = ml_algorithms.F1(probs7, Y_test7)
        F1_score7_repeat = ml_algorithms.F1(probs7_repeat, Y_test7_repeat)
        F1_score8 = ml_algorithms.F1(probs8, Y_test8)
        F1_score8_repeat = ml_algorithms.F1(probs8_repeat, Y_test8_repeat)
        
        F1_score1_mean = 0.5*(F1_score1+F1_score1_repeat)
        F1_score7_mean = 0.5*(F1_score7+F1_score7_repeat)
        F1_score8_mean = 0.5*(F1_score8+F1_score8_repeat)
        
        #calculate Kessler FoM values
        FoM1 = ml_algorithms.FoM(probs1, Y_test1)
        FoM1_repeat = ml_algorithms.FoM(probs1_repeat, Y_test1_repeat)
        FoM7 = ml_algorithms.FoM(probs7, Y_test7)
        FoM7_repeat = ml_algorithms.FoM(probs7_repeat, Y_test7_repeat)
        FoM8 = ml_algorithms.FoM(probs8, Y_test8)
        FoM8_repeat = ml_algorithms.FoM(probs8_repeat, Y_test8_repeat)
        
        FoM1_mean = 0.5*(FoM1+FoM1_repeat)
        FoM7_mean = 0.5*(FoM7+FoM7_repeat)
        FoM8_mean = 0.5*(FoM8+FoM8_repeat)
        
    #plot_probs(X_test, Y_test, probs)

    #calculate ROC curve values
    f2, t2, a2=ml_algorithms.roc(probs2, Y_test2)
    f2_repeat, t2_repeat, a2_repeat=ml_algorithms.roc(probs2_repeat, Y_test2_repeat)
    f3, t3, a3=ml_algorithms.roc(probs3, Y_test3)
    f3_repeat, t3_repeat, a3_repeat=ml_algorithms.roc(probs3_repeat, Y_test3_repeat)
    f4, t4, a4=ml_algorithms.roc(probs4, Y_test4)
    print("RF threshold = 0.5 fpr: %s" %(f4[250]))
    f4_repeat, t4_repeat, a4_repeat=ml_algorithms.roc(probs4_repeat, Y_test4_repeat)
    if probs5 != -9999:
        f5, t5, a5=ml_algorithms.roc(probs5, Y_test5)
        f5_repeat, t5_repeat, a5_repeat=ml_algorithms.roc(probs5_repeat, Y_test5_repeat)
    f6, t6, a6=ml_algorithms.roc(probs6, Y_test6)
    f6_repeat, t6_repeat, a6_repeat = ml_algorithms.roc(probs6_repeat, Y_test6_repeat)

    #calculate mean AUC over cross validation
    a2_mean = (a2+a2_repeat)/2.0
    a3_mean = (a3+a3_repeat)/2.0
    a4_mean = (a4+a4_repeat)/2.0
    #a5_mean = (a5+a5_repeat)/2.0
    a6_mean = (a6+a6_repeat)/2.0
    
    #calculate F1 values
    F1_score2 = ml_algorithms.F1(probs2, Y_test2)
    F1_score2_repeat = ml_algorithms.F1(probs2_repeat, Y_test2_repeat)
    F1_score3 = ml_algorithms.F1(probs3, Y_test3)
    F1_score3_repeat = ml_algorithms.F1(probs3_repeat, Y_test3_repeat)
    F1_score4 = ml_algorithms.F1(probs4, Y_test4)
    F1_score4_repeat = ml_algorithms.F1(probs4_repeat, Y_test4_repeat)
    if probs5 != -9999:
        F1_score5 = ml_algorithms.F1(probs5, Y_test5)
        F1_score5_repeat = ml_algorithms.F1(probs5_repeat, Y_test5_repeat)
    F1_score6 = ml_algorithms.F1(probs6, Y_test6)
    F1_score6_repeat = ml_algorithms.F1(probs6_repeat, Y_test6_repeat)
    
    F1_score2_mean = 0.5*(F1_score2+F1_score2_repeat)
    F1_score3_mean = 0.5*(F1_score3+F1_score3_repeat)
    F1_score4_mean = 0.5*(F1_score4+F1_score4_repeat)
    #F1_score5_mean = 0.5*(F1_score5+F1_score5_repeat)
    F1_score6_mean = 0.5*(F1_score6+F1_score6_repeat)
    
    #calculate Kessler FoM values
    FoM2 = ml_algorithms.FoM(probs2, Y_test2)
    FoM2_repeat = ml_algorithms.FoM(probs2_repeat, Y_test2_repeat)
    FoM3 = ml_algorithms.FoM(probs3, Y_test3)
    FoM3_repeat = ml_algorithms.FoM(probs3_repeat, Y_test3_repeat)
    FoM4 = ml_algorithms.FoM(probs4, Y_test4)
    FoM4_repeat = ml_algorithms.FoM(probs4_repeat, Y_test4_repeat)
    if probs5 != -9999:
        print("RNN WARNING")
        FoM5 = ml_algorithms.FoM(probs5,Y_test5)
        FoM5_repeat = ml_algorithms.FoM(probs5_repeat,Y_test5_repeat)
    FoM6 = ml_algorithms.FoM(probs6, Y_test6)
    FoM6_repeat = ml_algorithms.FoM(probs6_repeat, Y_test6_repeat)
    
    FoM2_mean = 0.5*(FoM2+FoM2_repeat)
    FoM3_mean = 0.5*(FoM3+FoM3_repeat)
    FoM4_mean = 0.5*(FoM4+FoM4_repeat)
    #FoM5_mean = 0.5*(FoM5+FoM5_repeat)
    FoM6_mean = 0.5*(FoM6+FoM6_repeat)
    
    print
    print 'AUC, F1, FoM:'
    if SVM:
        print 'SVM', a1_mean,  F1_score1_mean,  FoM1_mean
        print 'Cubic SVM', a7_mean,  F1_score7_mean,  FoM7_mean
        print 'RBF SVM',  a8_mean,  F1_score8_mean,  FoM8_mean
    print 'Bayes', a2_mean,  F1_score2_mean,  FoM2_mean
    print 'KNN', a3_mean,  F1_score3_mean,  FoM3_mean
    print 'Random forest', a4_mean,  F1_score4_mean,  FoM4_mean
   # print 'RNN',  a5, F1_score5_mean, FoM5_mean
    print 'AdaBoost forest',  a6_mean,  F1_score6_mean,  FoM6_mean
    

    figure(figsize=(10, 10))

    C1='#a21d21' #dark red
    C2='#185aa9' #blue
    C3='#f47d23' #orange
    C4='#008c48' #purple
    C5 ='#00b159' #green
    C6 ='#fd85ec' #pink
    C7 ='#a27e2c' #brown
    C8 ='#40e0d0' #cyan
    
    linew=2.5

    
    #plot ROC curves
    if SVM:
        plot(f1, t1, C1, lw=linew)
        plot(f7, t7, C7, lw=linew)
        plot(f8, t8, C8, lw=linew)
    plot(f2, t2, C2, lw=linew)
    plot(f3, t3, C3, lw=linew)
    plot(f4, t4, C4, lw=linew)
    #plot(f5, t5, C5, lw=linew)
    plot(f6, t6, C6, lw=linew)
    
    #Set plot parameters
    ax=gca()
    ax.set_aspect(1.0)
    setup_plot(ax)
    if SVM:
        legend(('SVM (%.3f)' %(a1), 'Cubic SVM (%.3f)' %(a7),  'RBF SVM (%.3f)' %(a8),  'Naive Bayes (%.3f)' %(a2), 'KNN (%.3f)' %(a3), 'Random Forest (%.3f)' %(a4), 
        'Ada Forest (%.3f)' %(a6)),  loc='lower right',  frameon=True, bbox_to_anchor=(0.95, 0.05), fontsize=18)
    else:
        legend(('Naive Bayes (%.3f)' %(a2), 'KNN (%.3f)' %(a3), 'Random Forest (%.3f)' %(a4), \
        'Ada Forest (%.3f)' %(a6)), loc='lower right',  frameon=True, bbox_to_anchor=(0.95, 0.05), fontsize=18)
        
    #legend(('Naive Bayes (%.3f)' %(a2), 'KNN (%.3f)' %(a3), 'Random Forest (%.3f)' %(a4),  'RRN (%.3f)' %(a5), 
    #'Ada Forest (%.3f)' %(a6)), loc='lower right',  frameon=True, bbox_to_anchor=(0.95, 0.05), fontsize=18)
    #legend(('Naive Bayes (%.3f)' %(a2), 'KNN (%.3f)' %(a3), 'Random Forest (%.3f)' %(a4)), loc='lower right')
    title('ROC Curve', fontsize=22)
    xlabel('False positive rate (contamination)', fontsize=18)
    ylabel('True positive rate (recall)', fontsize=18)
    
    subplots_adjust(bottom=0.08,left=0.05, top=0.92, right=0.95)
    show()
    
#plot_lc('Simulations/SIMGEN_PUBLIC_DES/DES_SN319694.DAT')
#plot_lc('Simulations/SIMGEN_PUBLIC_DES/DES_SN785053.DAT')
    

