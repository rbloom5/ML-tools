#!/usr/bin/env python
from __future__ import division
from sklearn.metrics import *
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from matplotlib.pyplot import loglog



def classif_metrics(evaluator, cv, X, y, precision_recall = False, auc = 0, pos_label = 1, predicted_ones = False, reduce_data = False):
    #Takes an sklearn ML evaluator object and a sklearn cv object and returns average scores of various metrics across folds
    #can only reduce data for ensemble methods (like random forest)
    #pos_label is the label for positive samples (1 is default)
    #set precision_recall, auc, and log_loss to True if you want those scores evaluated for each fold

    results = []
    precision = []
    recall = []
    F1 = []
    truepos = []
    falsepos = []
    auc_ = []
    num_ones = []

    if reduce_data is not False:
        evaluatorFit = evaluator.fit(X, y)
        if "n_estimators" in evaluatorFit.get_params():
            X = evaluatorFit.transform(X)


    for traincv, testcv in cv:

        evaluatorFit = evaluator.fit(X[traincv], y[traincv])
        #print evaluatorFit.coef_

        FoldScore = evaluatorFit.score(X[testcv], y[testcv])
        results.append(FoldScore)

        FoldPredicts = evaluatorFit.predict(X[testcv])
#         print FoldPredicts
#         print y[testcv]

        if precision_recall is not False:
            precision.append(precision_score(y[testcv], FoldPredicts))
            recall.append(recall_score(y[testcv], FoldPredicts)) 
            F1.append(f1_score(y[testcv], FoldPredicts)) 
            conf_mat = confusion_matrix(y[testcv],FoldPredicts)
            truepos.append(conf_mat[1][1])
            falsepos.append(conf_mat[0][1])

        if auc: 
            FoldProbas = evaluatorFit.predict_proba(X[testcv])
            Oneprobas = [x[1] for x in FoldProbas]
            auc_.append(roc_auc_score(y[testcv], Oneprobas))

        if predicted_ones is not False:
            predictions = evaluatorFit.predict(X[testcv])
            num_ones.append(predictions.sum())



    return {"score": np.array(results).mean(), \
            "precision": np.array(precision).mean(), \
            "recall": np.array(recall).mean(), \
            "F1": np.array(F1).mean(),\
            "truepos": np.array(truepos).mean(), \
            "falsepos": np.array(falsepos).mean(), \
            "auc": np.array(auc_).mean(),\
            "predicted_ones": np.array(num_ones).mean() }





def regression_metrics(evaluator, cv, X, y, precision_recall = False, auc = 0, pos_label = 1, predicted_ones = False, reduce_data = False):
    #Takes an sklearn ML evaluator object and a sklearn cv object and returns average scores of various metrics across folds
    #can only reduce data for ensemble methods (like random forest)
    #pos_label is the label for positive samples (1 is default)
    #set precision_recall, auc, and log_loss to True if you want those scores evaluated for each fold
    results = []
    precision = []
    recall = []
    F1 = []
    truepos = []
    falsepos = []
    auc_ = []
    num_ones = []

    if reduce_data:
        evaluatorFit = evaluator.fit(X, y)
        if "n_estimators" in evaluatorFit.get_params():
            X = evaluatorFit.transform(X)


    all_predicts = np.array([])
    all_actual = np.array([])
    for traincv, testcv in cv:


        evaluatorFit = evaluator.fit(X[traincv], y[traincv])
        #print evaluatorFit.coef_

#         FoldScore = evaluatorFit.score(X[testcv], y[testcv])
        results.append(evaluatorFit.score(X[testcv], y[testcv]))
        FoldPredicts = evaluatorFit.predict(X[testcv])
        
        #accumulate vectors of the predicted values from model and actual values in all folds
        all_predicts = np.concatenate((all_predicts, FoldPredicts), axis=0)
        all_actual = np.concatenate((all_actual, y[testcv]), axis = 0) 

        if precision_recall is not False:
            precision.append(precision_score(y[testcv], FoldPredicts))
            recall.append(recall_score(y[testcv], FoldPredicts)) 
            F1.append(f1_score(y[testcv], FoldPredicts)) 
            # conf_mat = confusion_matrix(y[testcv],FoldPredicts)
            # truepos.append(conf_mat[1][1]/(conf_mat[1][1]+conf_mat[1][0]))
            # falsepos.append(conf_mat[0][1]/(conf_mat[0][1]+conf_mat[1][1]))

        if auc: 
            FoldProbas = evaluatorFit.predict_proba(X[testcv])
            Oneprobas = [x[1] for x in FoldProbas]
            auc_.append(roc_auc_score(y[testcv], Oneprobas))

        if predicted_ones is not False:
            predictions = evaluatorFit.predict(X[testcv])
            num_ones.append(predictions.sum())
       
      
     #put all points in on vector to help with setting axes       
    all_pts = np.concatenate((all_predicts, all_actual), axis = 0)
    plt.figure()
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(all_actual, all_predicts)
    print 'r:', r_value, "r-squared:", r_value**2, ' p value: ', p_value, 'score', np.mean(np.array(results))
    
    ## PLOT ##
    plt.loglog( all_actual, all_predicts,'o', range(int(np.amax(all_pts))), range(int(np.amax(all_pts))), '-')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.xlim([0,np.amax(all_pts)])
    plt.ylim([0,np.amax(all_pts)])
    
#     plt.xscale('log')
#     plt.yscale('log')
    
    plt.show
    return {'r': r_value, "r-squared": r_value**2, ' p value': p_value, 'score': np.mean(np.array(results)) }

	        