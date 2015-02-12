from pylab import *
from sklearn import *
import max_ml_algorithms as ml_algorithms, pywt, os, math, time
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
    
    f2, t2, a2=ml_algorithms.bayes(X_train, Y_train, X_test, Y_test)
    f3, t3, a3=ml_algorithms.nearest_neighbours(X_train, Y_train, X_test, Y_test)
    f4, t4, a4=ml_algorithms.forest(X_train, Y_train, X_test, Y_test)
    f5, t5, a5=ml_algorithms.radius_neighbours(X_train, Y_train, X_test, Y_test)
    f6, t6, a6 = ml_algorithms.boost_RF(X_train, Y_train, X_test, Y_test)
    if SVM:
        f1, t1, a1=ml_algorithms.support_vm(X_train, Y_train, X_test, Y_test)
        f7, t7, a7=ml_algorithms.support_vm3(X_train, Y_train, X_test, Y_test)
    #plot_probs(X_test, Y_test, probs)

    print
    print 'AUC:'
    if SVM:
        print 'SVM',a1
        print 'Cubic SVM', a7
    print 'Bayes', a2
    print 'KNN', a3
    print 'Random forest', a4
    print 'RNN',  a5
    print 'AdaBoost forest',  a6
    

    figure(figsize=(10, 10))

    C1='#a21d21' #dark red
    C2='#185aa9' #blue
    C3='#f47d23' #orange
    C4='#008c48' #purple
    C5 ='#00b159' #green
    C6 ='#fd85ec' #pink
    C7 ='#a27e2c' #brown
    
    linew=2.5

    #plot ROC curves
    if SVM:
        plot(f1, t1, C1, lw=linew)
        plot(f7, t7, C7, lw=linew)
    plot(f2, t2, C2, lw=linew)
    plot(f3, t3, C3, lw=linew)
    plot(f4, t4, C4, lw=linew)
    plot(f5, t5, C5, lw=linew)
    plot(f6, t6, C6, lw=linew)
    
    #Set plot parameters
    ax=gca()
    ax.set_aspect(1.0)
    setup_plot(ax)
    if SVM:
        legend(('SVM (%.3f)' %(a1), 'Cubic SVM (%.3f)' %(a7),  'Naive Bayes (%.3f)' %(a2), 'KNN (%.3f)' %(a3), 'Random Forest (%.3f)' %(a4), 'RNN (%.3f)' %(a5),  'Ada Forest (%.3f)' %(a6)),  loc='lower right',  frameon=True, bbox_to_anchor=(0.95, 0.05), fontsize=18)
    else:
        legend(('Naive Bayes (%.3f)' %(a2), 'KNN (%.3f)' %(a3), 'Random Forest (%.3f)' %(a4),  'RRN (%.3f)' %(a5),  'Ada Forest (%.3f)' %(a6)), loc='lower right',  frameon=True, bbox_to_anchor=(0.95, 0.05), fontsize=18)
    #legend(('Naive Bayes (%.3f)' %(a2), 'KNN (%.3f)' %(a3), 'Random Forest (%.3f)' %(a4)), loc='lower right')
    title('ROC Curve', fontsize=22)
    xlabel('False positive rate (contamination)', fontsize=18)
    ylabel('True positive rate (recall)', fontsize=18)
    
    subplots_adjust(bottom=0.08,left=0.05, top=0.92, right=0.95)
    show()
    
#plot_lc('Simulations/SIMGEN_PUBLIC_DES/DES_SN319694.DAT')
#plot_lc('Simulations/SIMGEN_PUBLIC_DES/DES_SN785053.DAT')
    

