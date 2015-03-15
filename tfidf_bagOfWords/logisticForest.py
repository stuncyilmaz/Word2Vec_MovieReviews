import numpy as np

import pandas as pd
from sklearn import preprocessing

import sklearn 
from  sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer


def resample(X,y=None,random_state=None,return_unsampled=False,replace=True):
    """
    Create a bootstrap sample of the data of the same size.
    :param data: list of any format you want! Usually a dataset for ML
    :return: Tuple: (Bootstrap sample, OOB data)
    """
    if random_state!=None:
        np.random.seed(random_state)
    N = X.shape[0]
    selected_ind=np.random.choice(range(N), size=N, replace=replace)
    not_selected_ind=[elt for elt in range(N)    if elt  not in selected_ind]
    X_new=X[selected_ind]
    
    if not return_unsampled:
        X_unsampled=None
    else:
        X_unsampled=X[not_selected_ind]

    if not np.any(pd.isnull(y)):
        y=np.array(y)
        y_new=y[selected_ind]
        if not return_unsampled:
            y_unsampled=None
        else:
            y_unsampled=y[not_selected_ind]    
        return {"X_new":X_new,"X_unsampled":X_unsampled,"y_new":y_new,"y_unsampled":y_unsampled}
    return {"X_new":X_new,"X_unsampled":X_unsampled}


class logisticForest(sklearn.base.BaseEstimator):
    
    def __init__(self,n_folds=100,feature_size=10,clf = LogisticRegression()):
        self.clf = clf 
        self.n_folds=n_folds # number of classifiers
        self.classifiers=[sklearn.base.clone(clf) for i in range(self.n_folds)]
        self.scores=[None]*self.n_folds
        self.N=None
        self.lb=LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
        self.kf_features=None   # indices for selected features
        self.feature_size=feature_size # number of features to consider
            
    
    def fit(self,X,y):
        
        if self.feature_size<1:
            self.feature_size=np.max([1,int(self.feature_size*X.shape[1])])

        self.kf_features=[None]*self.n_folds
        for i in range (self.n_folds):
            selected_ind=np.random.choice(range(X.shape[1]), size=self.feature_size, replace=False)
            self.kf_features[i]=selected_ind
        self.N=X.shape[0]
        
        self.scores=[None]*self.n_folds
        self.lb.fit(y)
        y=self.lb.transform(y).reshape(-1,)


        for j,(features_idx) in enumerate(self.kf_features):
            data_current=resample(X,y,return_unsampled=True)
            X_current=data_current['X_new'][:,features_idx]
            y_current=data_current['y_new'][:]
            
            X_current_test=data_current["X_unsampled"][:,features_idx]
            y_current_test=data_current['y_unsampled'][:]
            
            clf_current=self.classifiers[j]
            clf_current.fit(X_current,y_current)
            self.scores[j]=roc_auc_score(y_current_test,clf_current.predict_proba(X_current_test)[:,1])
            if j%10==0: 
                print ('iteration %i, score: %0.2f')%(j,self.scores[j])
        self.scores=np.array(self.scores)
        self.scores=self.scores-np.mean(self.scores)
        self.scores=self.scores/(np.max(self.scores)-np.min(self.scores))
        self.scores=self.scores+np.abs(np.min(self.scores))
        self.scores/=np.sum(self.scores)

    
    def predict_proba(self,X,weighted=False,bootstrap=False):
        predictions=[None]*self.n_folds
        if not weighted:
            my_scores=np.array(([1]*self.n_folds))/float(self.n_folds)
        else:
            my_scores=self.scores
        for j,(features_idx) in enumerate(self.kf_features):

            X_current=X[:,features_idx]
            predictions[j]=self.classifiers[j].predict(X_current)*my_scores[j]
        predictions=np.array(predictions)
        return np.sum(predictions,axis=0)
    
    def predict(self,X,weighted=False):
        predictions=self.predict_proba(X,weighted)
        return np.where(predictions>0.5,1,0)
            
    def Score_AUC(self,X,y,weighted=False):
        y=self.lb.transform(y).reshape(-1,)
        return roc_auc_score(y,self.predict_proba(X,weighted))
        
