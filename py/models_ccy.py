#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 06:54:38 2017

@author: changyaochen
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# let's handle everything in a class
class Classifier():
    '''
    In this class, we will do everything, including:
    (1) data cleaning, imputation, scaling
    (2) possible feature engineering
    (3) model fitting, including CV 
    (4) prediction on test data
    (5) helper functions such as various plotting functions
    
    Here we heavily rely on the pandas, hence most of the datatypes is pandas dataframe
    '''
    def __init__(self, X_raw, y_raw):
        '''
        Initialization.
        Here we assume there is no missing entry (with value of -1)
        in the data, i.e., we've done the data cleaning step.
        
        Input
        =====================
        X_raw: <pandas dataframe> 
            The raw input, in pandas dataframe. 
            The shape should be m x n,
            where m is the number of sample, n is the number of features
        Y_raw: <pandas dataframe>
            The raw binary labels, in pandas dataframe.
            The shape should be m x 1
            
        Return
        =====================
        Nothing
        '''
        import time
        
        self.X_raw = X_raw
        self.y_raw = y_raw
        self.one_hot = True
        # check for missing values
        tmp_cnt = 0
        tmp_cols = []
        for col in X_raw.columns:
            if X_raw[col].min() < 0:
                tmp_cnt += 1
                tmp_cols.append(col)
        if tmp_cnt > 0:
            print('Thre are {} features with missing value(s)'.format(tmp_cnt))
            print('They are:')
            for x in tmp_cols:
                print('    ', x)
            print('======= Please deal with the missing values first! =======\n')
    
    def plot_roc(self, fpr, tpr, roc_auc, model_type='not specified'):
        """
        Helper function. 
        To plot the ROC curve for a binary classifier.
        
        Input:
        =========
        fpr: <float>
            false positive rate
        tpr: <float>
            true positive rate
        roc_auc: <???>
            ???
            
        Return:
        =========
        Nothing, draw a plot.
        """        
        
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = {:0.6f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for {}'.format(model_type))
        plt.legend(loc="lower right")
        plt.show()
    
    def gini(self, actual, pred):
        '''
        Calculate the gini coefficient for a binary classifer.
        Copied from 
        https://www.kaggle.com/batzner/gini-coefficient-an-intuitive-explanation
        
        Input
        ==========
        actual: <list> 
            a list of 0/1 that represent the actual labels
        pred: <list>
            a list of floats between 0 and 1 that are the predicted probability
            
        Return
        ==========
        <float>
            the calculated gini coefficient.
        '''
        assert (len(actual) == len(pred))
        all_tmp = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
        all_tmp = all_tmp[np.lexsort((all_tmp[:, 2], -1 * all[:, 1]))]
        totalLosses = all_tmp[:, 0].sum()
        giniSum = all_tmp[:, 0].cumsum().sum() / totalLosses

        giniSum -= (len(actual) + 1) / 2.
        return giniSum / len(actual)
    
    def ini_normalized(self, a, p):
        return self.gini(a, p) / self.gini(a, a)
    
    def get_importance(self, clf, X, N=3):
        """
        For the Random Forest and XGB, to get the feature importance
        
        TODO: finish the doc
        """
        features = X.columns
        importances = [[c, i] for c, i in zip(features, clf.feature_importances_)];
        importances = sorted(importances, key=lambda x: x[1], reverse = True);
        N = 3
        print('\nThe top {} important features are:'.format(N))
        for j in range(N):
            print(importances[j][0], importances[j][1])     
    
    def input_scaling(self, ratio=0.2):
        '''
        First split the total data into training and testing, with given ratio
        and the apply the standard scaler to the training data
        
        Input
        ================
        ratio: <float>, default: 0.2
            ratio between test and training data sizes
        
        Return
        ================
        X_train: <pandas dataframe> 
            scaled training X
        y_train: <pandas dataframe>
            labels for the training X
        X_test: <pandas dataframe>
            scaled test X
        y_test: <pandas dataframe>
            lables for test X
        '''
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        #from sklearn.model_selection import GridSearchCV
        #from sklearn.metrics import confusion_matrix

        self.X_train, self.X_test, self.y_train, self.y_test \
            = train_test_split(self.X_raw, self.y_raw, test_size=ratio)
        # scaling the features
        scaler = StandardScaler()
        scaler.fit(self.X_train)  # Don't cheat - fit only on training data
        
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)  # apply same transformation to test data
        print('Splitted training/test, and applied standard scaler.')
        print('Total number of training samples: {}'.format(len(self.X_train)))
        
    def logistic_regression(self, tuning=True, roc_plot=True):
        '''
        Build a logistic regression model.
        
        Input:
        ======
        tuning: <Boolean> default: True
            If True, perform the hyper parameter tuning, with GridSearch CV
        roc_plot: <Boolean> default: True
            If True, plot the roc plot and calculate AUC
        
        Return:
        =======
        The built logistic regression classifier.
        '''
        from sklearn.model_selection import GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_curve, auc
        
        print('......Training a logistic regression model......')
        if not self.one_hot:
            print('We haven\'t done one hot encoding yet! Quit training.')
            return
        # get the single classifier
        self.lg_clf = LogisticRegression(class_weight='balanced')
        # hyper parameters
        parameters = {'C': np.logspace(-3, 0, num=30)}
        
        if tuning:  # hyper parameter tuning
            lg_clfs = GridSearchCV(self.lg_clf, parameters, 
                                   verbose = 1, cv = 5, n_jobs = -1)
            lg_clfs.fit(self.X_train, self.y_train)
            self.lg_clf = lg_clfs.best_estimator_
        else:
            self.lg_clf.fit(self.X_train, self.y_train)
        
        pred = self.lg_clf.predict_proba(self.X_test)[:,1]  # prob of class 1
        print('The training score is: {}'.format(self.lg_clf.score(self.X_train, self.y_train)))
        print('The test score is: {}'.format(self.lg_clf.score(self.X_test, self.y_test)))
        print('The gini coefficient is : {}'.format(
            self.gini_normalized(self.y_test.values, pred)))
        
        if roc_plot:
            ## get the roc curve
            y_score = self.lg_clf.decision_function(self.X_test)
            fpr, tpr, _ = roc_curve(self.y_test, y_score, drop_intermediate=False)
            roc_auc = auc(fpr, tpr)
            self.plot_roc(fpr, tpr, roc_auc, model_type='Logistic Regression')
        return self.lg_clf
            
    def random_forest(self, tuning=True, roc_plot=True):
        '''
        Build a random forest model.
        
        Input:
        ======
        tuning: <Boolean> default: True
            If True, perform the hyper parameter tuning, with GridSearch CV
        roc_plot: <Boolean> default: True
            If True, plot the roc plot and calculate AUC
        
        Return:
        =======
        The built random forest classifier.
        '''
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import roc_curve, auc
        
        print('......Training a random forest model......')
        if not self.one_hot:
            print('We haven\'t done one hot encoding yet! Quit training.')
            return
        parameters = {'max_depth': [x for x in range(3, 21, 1)],
              'n_estimators':[x for x in range(100, 401, 100)]}
        self.rf_clf = RandomForestClassifier(class_weight='balanced',
                                             verbose = False, random_state = True)
        if tuning:
            rf_clfs = GridSearchCV(self.rf_clf, parameters, 
                                   verbose = 1, cv = 5, n_jobs = -1)
            rf_clfs.fit(self.X_train, self.y_train)
            self.rf_clf = rf_clfs.best_estimator_
        else:
            self.rf_clf.fit(self.X_train, self.y_train)
        
        pred = self.rf_clf.predict_proba(self.X_test)
        print('The training score is: {}'.format(self.rf_clf.score(self.X_train, self.y_train)))
        print('The test score is: {}'.format(self.rf_clf.score(self.X_test, self.y_test)))
        print('The gini coefficient is : {}'.format(self.gini_normalized(self.y_test, pred)))
        
        if roc_plot:
            ## get the roc curve
            y_score = self.rf_clf.predict_proba(self.X_test)
            y_score = [x[1] for x in y_score]
            fpr, tpr, _ = roc_curve(self.y_test, y_score, drop_intermediate=False)
            roc_auc = auc(fpr, tpr)
            self.plot_roc(fpr, tpr, roc_auc, model_type='Random Forest')
        return self.rf_clf

# ===== below are copied from jjf's code =====      
class my_xgb():
    def __init__(self, X_raw, y_raw):
        # Set xgboost parameters
        self.params = {}
        self.params['objective'] = 'binary:logistic'
        self.params['eta'] = 0.05
        self.params['silent'] = True
        self.params['max_depth'] = 6
        self.params['subsample'] = 0.8
        self.params['colsample_bytree'] = 0.8
        self.params['eval_metric'] = 'auc'
        self.params['max_delta_step'] = 1
        
        self.X_raw = X_raw
        self.y_raw = y_raw
        
        # set some check flags
        self.scaling_flag = False
        self.one_hot_flag = False
    
    # Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
    def gini(self, actual, pred, cmpcol = 0, sortcol = 1):
        assert( len(actual) == len(pred) )
        all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
        all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
        totalLosses = all[:,0].sum()
        giniSum = all[:,0].cumsum().sum() / totalLosses
        
        giniSum -= (len(actual) + 1) / 2.
        return giniSum / len(actual)
     
    def gini_normalized(self, a, p):
        return self.gini(a, p) / self.gini(a, a)
    
    # Create an XGBoost-compatible metric from Gini
    
    def gini_xgb(self, preds, dtrain):
        labels = dtrain.get_label()
        gini_score = self.gini_normalized(labels, preds)
        return [('gini', gini_score)]
    
    def input_scaling(self, ratio=0.2):
        '''
        First split the total data into training and testing, with given ratio
        and the apply the standard scaler to the training data
        
        Input
        ================
        X_raw: <pd dataframe>
            raw input, without scaling, but no missing values
        Y_raw: <pd dataframe>
            labels for the X_raw
        ratio: <float>, default: 0.2
            ratio between test and training data sizes
        
        Return
        ================
        X_train: <pandas dataframe> 
            scaled training X
        y_train: <pandas dataframe>
            labels for the training X
        X_test: <pandas dataframe>
            scaled test X
        y_test: <pandas dataframe>
            labels for test X
        '''
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        #from sklearn.model_selection import GridSearchCV
        #from sklearn.metrics import confusion_matrix
        
        cols = self.X_raw.columns  
        self.X_train, self.X_test, self.y_train, self.y_test \
            = train_test_split(self.X_raw, self.y_raw, test_size=ratio)
        # scaling the features
        scaler = StandardScaler()
        scaler.fit(self.X_train)  # Don't cheat - fit only on training data
        
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)  # apply same transformation to test data
        print('Splitted training/test, and applied standard scaler.')
        print('Total number of training samples: {}'.format(len(self.X_train)))
        self.X_train = pd.DataFrame(self.X_train, columns=cols)
        self.y_train = pd.DataFrame(self.y_train)
        self.X_test = pd.DataFrame(self.X_test, columns=cols)
        self.y_test = pd.DataFrame(self.y_test)

        self.scaling_flag = True

    def one_hot_encoding(self, X_train=None, X_test=None):
        '''
        One hot eocoding the features with _cat suffix

        Input
        ============
        X_train: <pd dataframe>, default: None
            The non-one-hot-encoded dataframe. If None, use the inherent one.
        X_test: <pd dataframe>, default: None
            The non-one-hot-encoded dataframe for test. If None, use the inherent one.

        Return
        ============
        Nothing, but update self.X_train and self.X_test
        '''
        if X_train is None:
            X_train = self.X_train
        if X_test is None:
            X_test = self.X_test
        print('Shape of dataframe before one hot encoding is {}.'.format(X_train.shape))
        
        to_one_hot = []
        idx = X_train.shape[0]  # boundary
        X_tot = pd.concat((X_train, X_test))  # put the together so that one hot encode on both of them
        for col in X_tot.columns:
            if col.endswith('_cat'):
                to_one_hot.append(col)
        X_tot = pd.get_dummies(X_tot, columns=to_one_hot)
        self.X_train = X_tot.iloc[:idx, :]
        self.X_test = X_tot.iloc[idx:, :]
        self.one_hot_flag = True
        print('Done one hot encoding on {} features'.format(len(to_one_hot)))
        print('Shape of dataframe after one hot encoding is {}.'.format(self.X_train.shape))

  
    def fit(self, X_train=None, y_train=None, K=5, params=None, scaling=True, one_hot=True):
        '''
        perform the k-fold cross validation to train the xgboost model

        Input
        ============
        X_train: <pd dataframe>, default: None
            A pandas dataframe that contains all the features. If None, use the inherent one.
        y_train: <pd dataframe>, default: None
            A pandas dataframe that contains all labels. If None, use the inherent one.
        K: <int>, default: 5
            number of folds
        params: <dict>, defautl: None
            parameters for the xgboost model. If None, then use the inherent one.
        scaling: <Boolean>, default: True
            If True, requires the dataset is standardized.
        one_hot: <Boolean>, default: True
            If True, requires the dataset is one_hot_encoded. 

        Return
        ============
        xgb_models: <list>
            A list of K models, that are trained on different n_splits
        xgb_metrics: <list>
            A list of dictionaries that contains the evaluation metrics on the train and validation set
        '''

        import xgboost as xgb 
        from sklearn.model_selection import StratifiedKFold
        
        if scaling and not self.scaling_flag:
            print('The inputs are not scaled, stop training.')
            return
        if one_hot and not self.one_hot_flag:
            print('The categorical features are not one hot encoded, stop training.')
            return
        
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        if params is None:
            params = self.params
        
        xgb_models = []
        xgb_metrics = []
        xgb_preds = []  # currently not return
        
        target_train = y_train.values.squeeze()
        train = np.array(X_train.values)
        test = np.array(self.X_test.values)
        # k-fold CV
        K = 5
        kf = StratifiedKFold(n_splits=K, random_state=3228, shuffle=True)
        
        print('Performing {} fold CV'.format(K))
        fold = 1
        for train_index, test_index in kf.split(train, target_train):
            print('Fold {} of {}'.format(fold, K))
            train_X, valid_X = train[train_index], train[test_index]
            train_y, valid_y = target_train[train_index], target_train[test_index]
        
            d_train = xgb.DMatrix(train_X, train_y)
            d_valid = xgb.DMatrix(valid_X, valid_y)
            d_test = xgb.DMatrix(test)
            
            # update the class weight
            sum_wpos = (train_y == 1).sum()
            sum_wneg = (train_y == 0).sum()
            self.params['scale_pos_weight'] = 1.* sum_wneg/sum_wpos
            print('class weight is: {}'.format(sum_wneg/sum_wpos))
            
            watchlist = [(d_train, 'train'), (d_valid, 'valid')]
            evals_result = {}
            self.model = xgb.train(params, 
                                  d_train, 
                                  5000,  
                                  watchlist, 
                                  feval=self.gini_xgb, 
                                  maximize=True, 
                                  verbose_eval=100, 
                                  early_stopping_rounds=100,
                                  evals_result=evals_result)
            # make prediction!
            xgb_pred = self.model.predict(d_test)
            xgb_preds.append(list(xgb_pred))
            
            # prepare the return values
            xgb_models.append(self.model)
            # metrics = evals_result[watchlist[0][-1]].keys()
            xgb_metrics.append(evals_result)

            fold += 1
        return xgb_models, xgb_metrics


