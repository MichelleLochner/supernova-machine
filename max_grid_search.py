import max_ml_algorithms as ml_algorithms
import numpy as np
import time


def KNN_optimiser(X_train, Y_train, X_test, Y_test, param_dict):
    """
    Implements a grid search parameter optimisation of the KNN classifier with respect
    to ROC AUC.
    
    INPUTS:
    X_train - An array containing the features of the training set, of size (N_samples, N_features)
    Y_train - An array containing the class labels of the training set, of size (N_samples,)
    X_test - An array containing the features of the testing set, of size (N_samples, N_features)
    Y_test - An array containing the class labels of the testing set, of size (N_samples,)
    param_dict - A dictionary containing the parameter values to be searched over
    
    OUTPUTS:
    best_params - A dictionary containing the optimum parameter values
    """
    
    start = time.time()

    #Add edge elements to parameter grid
    lower_edge = int(round(param_dict['n_neighbors'][0]*0.9))
    upper_edge = int(round(param_dict['n_neighbors'][-1]*1.1))
    param_dict['n_neighbors'].insert(0,lower_edge)
    param_dict['n_neighbors'].append(upper_edge)

    
    row_counter = 0
    col_counter = 0
    auc_scores = np.zeros((len(param_dict['n_neighbors']), len(param_dict['weights'])))

    for N_counter in param_dict['n_neighbors']:
        for W_counter in param_dict['weights']:
            
            probs=ml_algorithms.nearest_neighbours(X_train, Y_train, X_test, N_counter, W_counter)
            fpr,  tpr,  auc_scores[row_counter, col_counter] = ml_algorithms.roc(probs, Y_test)
            
            col_counter += 1
        
        col_counter = 0
        row_counter += 1

    K_index, W_index = np.unravel_index(np.argmax(auc_scores), auc_scores.shape)

    #Check if K_index is lower_edge or upper_edge
    if param_dict['n_neighbors'][K_index]==lower_edge:
        K_index += 1
        print('Lower bound on KNN N_neighbor range may be too high \n')
    if param_dict['n_neighbors'][K_index]==upper_edge:
        K_index -= 1
        print('Upper bound on KNN N_neighbor range may be too low \n')
    
    best_params = {'n_neighbors':param_dict['n_neighbors'][K_index], 'weights':param_dict['weights'][W_index]}
    
    print("KNN optimiser time taken: %s" %(time.time()-start))
    
    return best_params
    
    
    
    
    
def RF_optimiser(X_train, Y_train, X_test, Y_test, param_dict):
    """
    Implements a grid search parameter optimisation of the RF classifier with respect to 
    ROC AUC.
    
    INPUTS:
    X_train - An array containing the features of the training set, of size (N_samples, N_features)
    Y_train - An array containing the class labels of the training set, of size (N_samples,)
    X_test - An array containing the features of the testing set, of size (N_samples, N_features)
    Y_test - An array containing the class labels of the testing set, of size (N_samples,)
    param_dict - A dictionary containing the parameter values to be searched over
    
    OUTPUTS:
    best_params - A dictionary containing the optimum parameter values
    """
    
    start = time.time()
    
    #Add edge elements to parameter grid
    lower_edge = int(round(param_dict['n_estimators'][0]*0.9))
    upper_edge = int(round(param_dict['n_estimators'][-1]*1.1))
    param_dict['n_estimators'].insert(0,lower_edge)
    param_dict['n_estimators'].append(upper_edge)

    row_counter = 0
    col_counter = 0
    auc_scores = np.zeros((len(param_dict['n_estimators']), len(param_dict['criterion'])))
    
    for N_counter in param_dict['n_estimators']:
        for C_counter in param_dict['criterion']:
            
            probs=ml_algorithms.forest(X_train, Y_train, X_test, N_counter, C_counter)
            fpr,  tpr,  auc_scores[row_counter, col_counter] = ml_algorithms.roc(probs, Y_test)
            
            col_counter += 1
        
        col_counter = 0
        row_counter += 1
    
    N_index, C_index = np.unravel_index(np.argmax(auc_scores), auc_scores.shape)
    
    #Check if N_index is lower_edge or upper_edge
    if param_dict['n_estimators'][N_index]==lower_edge:
        N_index += 1
        print('Lower bound on RF N_estimators range may be too high \n')
    if param_dict['n_estimators'][N_index]==upper_edge:
        N_index -= 1
        print('Upper bound on RF N_estimators range may be too low \n')

    best_params = {'n_estimators':param_dict['n_estimators'][N_index], 'criterion':param_dict['criterion'][C_index]}
    
    print("RF optimisier time taken: %s" %(time.time()-start))
    
    return best_params
    
    
    
def Boost_optimiser(X_train, Y_train, X_test, Y_test, param_dict):
    """
    Implements a grid search parameter optimisation of the Boost classifier with respect to 
    ROC AUC.
    
    INPUTS:
    X_train - An array containing the features of the training set, of size (N_samples, N_features)
    Y_train - An array containing the class labels of the training set, of size (N_samples,)
    X_test - An array containing the features of the testing set, of size (N_samples, N_features)
    Y_test - An array containing the class labels of the testing set, of size (N_samples,)
    param_dict - A dictionary containing the parameter values to be searched over
    
    OUTPUTS:
    best_params - A dictionary containing the optimum parameter values
    """
    
    start = time.time()

    #Add edge elements to parameter grid
    lower_edge = int(round(param_dict['n_estimators'][0]*0.9))
    upper_edge = int(round(param_dict['n_estimators'][-1]*1.1))
    param_dict['n_estimators'].insert(0,lower_edge)
    param_dict['n_estimators'].append(upper_edge)
    
    row_counter = 0
    col_counter = 0
    auc_scores = np.zeros((len(param_dict['base_estimator']), len(param_dict['n_estimators'])))
    
    for B_counter in param_dict['base_estimator']:
        for N_counter in param_dict['n_estimators']:
            
            probs=ml_algorithms.boost_RF(X_train, Y_train, X_test, B_counter, N_counter)
            fpr,  tpr,  auc_scores[row_counter, col_counter] = ml_algorithms.roc(probs, Y_test)
            
            col_counter += 1
        
        col_counter = 0
        row_counter += 1
    
    B_index, N_index = np.unravel_index(np.argmax(auc_scores), auc_scores.shape)
    
    #Check if N_index is lower_edge or upper_edge
    if param_dict['n_estimators'][N_index]==lower_edge:
        N_index += 1
        print('Lower bound on RF N_estimators range may be too high \n')
    if param_dict['n_estimators'][N_index]==upper_edge:
        N_index -= 1
        print('Upper bound on RF N_estimators range may be too low \n')

    best_params = {'base_estimator':param_dict['base_estimator'][B_index], 'n_estimators':param_dict['n_estimators'][N_index]}
    
    print("AdaBoost optimiser time taken: %s" %(time.time()-start))
    
    return best_params
    


def RBF_optimiser(X_train, Y_train, X_test, Y_test, param_dict):
    """
    Implements a grid search parameter optimisation of the SVM classifier with RBF kernel
    with respect to ROC AUC.
    
    INPUTS:
    X_train - An array containing the features of the training set, of size (N_samples, N_features)
    Y_train - An array containing the class labels of the training set, of size (N_samples,)
    X_test - An array containing the features of the testing set, of size (N_samples, N_features)
    Y_test - An array containing the class labels of the testing set, of size (N_samples,)
    param_dict - A dictionary containing the parameter values to be searched over
    
    OUTPUTS:
    best_params - A dictionary containing the optimum parameter values
    """
    
    start = time.time()

    #Add edge elements to parameter grid
    lower_edge_C = param_dict['C'][0]*0.9
    upper_edge_C = param_dict['C'][-1]*1.1
    param_dict['C'].insert(0,lower_edge_C)
    param_dict['C'].append(upper_edge_C)
    lower_edge_g = param_dict['gamma'][0]*0.9
    upper_edge_g = param_dict['gamma'][-1]*1.1
    param_dict['gamma'].insert(0,lower_edge_g)
    param_dict['gamma'].append(upper_edge_g)
    
    row_counter = 0
    col_counter = 0
    auc_scores = np.zeros((len(param_dict['C']), len(param_dict['gamma'])))
    
    for C_counter in param_dict['C']:
        for G_counter in param_dict['gamma']:
            
            probs=ml_algorithms.support_vmRBF(X_train, Y_train, X_test, C_counter , G_counter)
            fpr,  tpr,  auc_scores[row_counter, col_counter] = ml_algorithms.roc(probs, Y_test)
            
            col_counter += 1
        
        col_counter = 0
        row_counter += 1
    
    C_index, G_index = np.unravel_index(np.argmax(auc_scores), auc_scores.shape)
    
    #Check if N_index is lower_edge or upper_edge
    if param_dict['C'][C_index]==lower_edge_C:
        C_index += 1
        print('Lower bound on Boost C range may be too high \n')
    if param_dict['C'][C_index]==upper_edge_C:
        C_index -= 1
        print('Upper bound on Boost C range may be too low \n')
    if param_dict['gamma'][G_index]==lower_edge_g:
        G_index += 1
        print('Lower bound on Boost gamma range may be too high \n')
    if param_dict['gamma'][G_index]==upper_edge_g:
        G_index -= 1
        print('Upper bound on Boost gamma range may be too low \n')

    best_params = {'C':param_dict['C'][C_index], 'gamma':param_dict['gamma'][G_index]}
    
    print("RBF SVM optimiser time taken: %s" %(time.time()-start))
    
    return best_params
    
    
    
