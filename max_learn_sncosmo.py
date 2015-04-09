"""
The main script. This splits data into training and testing sets, runs the classifiers on 
those sets, and prints the results. 

INPUTS:
- A feature set file with white space separated values, and comments delimited with #
  the columns containing features and errors must be specified below
  
OUTPUTS:
- The average and std of the optimum probability thresholds are printed to screen
- The average and std of the optimum AUC, F1 and FoM are printed to screen

"""

import numpy as np
from sklearn import preprocessing
import max_ml_algorithms as ml_algorithms, max_tools as tools
from sklearn.cross_validation import train_test_split
import time

start = time.time()

feats = np.genfromtxt('sncosmo_des_fit_emcee.txt', dtype='str', comments='#')
#errors = np.genfromtxt('sncosmo_des_fit_emcee.txt', dtype='str', comments='#')

#Features
f = np.array(feats[:, [4, 6, 7, 8]], dtype='float')
err = np.array(feats[:, [9, 11, 12, 13]], dtype='float')
types = np.array(feats[:, 1], dtype='int')

#To use fake features with known distribution for testing, uncomment next line
f, err, types = tools.make_a_fake_dataset()

X=f[types!=-9, :]
Y=types[types!=-9]
X_err = err[types!=-9, :]

X_scaled, X_err_scaled = tools.scale_data_with_errors(X, X_err)

#Combine features and errors for ease of splitting into train and test sets
X_combined = np.concatenate((X_scaled, X_err_scaled), axis=1)

#Combine classes into Ia (Y=1), II (Y=2), and Ib and Ic (Y=3)
Y[(Y==21) | (Y==22) | (Y==23)]=2
Y[(Y==32) | (Y==33)]=3

#Preallocate results table for multiple runs
#NOTE I'M HARD CODING THE NUMBER OF CLASSIFIERS AND PERFORMANCE
#CRITERION HERE
N_runs = 1
N_classifiers = 7
N_scores = 3
results = -999*np.ones([N_classifiers, N_scores, N_runs])
thresholds = -999*np.ones([N_classifiers, N_scores-1, N_runs])

#Run classifiers and return AUC, FoM and F1 for each
for run_counter in np.arange(N_runs):
   
    X_train_combined, X_test_combined, Y_train, Y_test = train_test_split(
    X_combined, Y, test_size=0.5, random_state=np.random.randint(100))
    
    #Seperate features and errors again
    X_train = X_train_combined[:, :X_scaled.shape[1]]
    X_train_err = X_train_combined[:, X_scaled.shape[1]:]
    X_test = X_test_combined[:, :X_scaled.shape[1]]
    X_test_err = X_test_combined[:, X_scaled.shape[1]:]
    
    #Run classifiers
    results_temp, thresholds_temp = tools.run_ml(X_train, Y_train, X_test, Y_test, X_train_err, X_test_err)
    results[:, :, run_counter] = results_temp
    thresholds[:, :, run_counter] = thresholds_temp
    
    #Clean up a bit
    del X_train, X_test, Y_train, Y_test, results_temp, thresholds_temp, X_train_combined, X_test_combined

#Calculate some performance stats
sigma_results = np.std(results, axis=2)
mean_results = np.mean(results, axis=2)

sigma_thresholds = np.std(thresholds, axis=2)
mean_thresholds = np.mean(thresholds, axis=2)

#Columns are in order F1 threshold, FoM threshold
#rows are RBF, KNN, RF, Boost, ANN, MCS
print("Average thresholds are: \n")
print(mean_thresholds)
print
print("Standard deviation of thresholds are: \n")
print(sigma_thresholds)
print

#Columns are in order AUC, F1, FoM, and rows are 
#RBF, NB, KNN, RF, Boost, ANN, MCS
print("Average results are: \n")
print(mean_results)
print
print("Standard deviation of results are: \n")
print(sigma_results)
print

#Find the best classifier wrt to each performance criterion
classifier_list = ['RBF', 'NB', 'KNN', 'RF', 'BOOST', 'ANN', 'MCS']
AUC_best = np.argmax(mean_results[:, 0])
F1_best = np.argmax(mean_results[:, 1])
FoM_best = np.argmax(mean_results[:, 2])

print('Best classifier by AUC is: %s' %(classifier_list[AUC_best]))
print('Best classifier by F1 is: %s' %(classifier_list[F1_best]))
print('Best classifier by FoM is: %s' %(classifier_list[FoM_best]))

print "Total run time: ",  time.time()-start


















    
