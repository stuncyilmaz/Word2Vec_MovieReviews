from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
def getCrossValidation(X_all_train,y,clf,params):

    test_scores=[[None]*len(params)]*5
    train_scores=[[None]*len(params)]*5
    test_prob_scores=[[None]*len(params)]*5
    train_prob_scores=[[None]*len(params)]*5

    kf = cross_validation.KFold(X_all_train.shape[0], n_folds=5,shuffle=True)
    for j,(train_idx, test_idx) in enumerate(kf):
        print('validation set: %i'%(j))
        X_train=X_all_train[train_idx]
        X_test=X_all_train[test_idx]
        y_train=y[train_idx]
        y_test=y[test_idx]
        for i,p in enumerate(params):
            clf.set_params(C=p)

            clf.fit(X_train,y_train)

            test_prob_scores[j][i]=roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
            train_prob_scores[j][i]=roc_auc_score(y_train,clf.predict_proba(X_train)[:,1])

            test_scores[j][i]=clf.score(X_test,y_test)
            train_scores[j][i]=clf.score(X_train,y_train)
    test_score=np.max(np.mean(np.array(test_scores),axis=0))
    train_score=np.max(np.mean(np.array(train_scores),axis=0))
    test_auc=np.max(np.mean(np.array(test_prob_scores),axis=0))
    train_auc=np.max(np.mean(np.array(train_prob_scores),axis=0))
    
    print('')
    print("cross-validation accuracy:%.3f"%(test_score))
    print("training accuracy:%.3f"%(train_score))
    print("cross-validation AUC:%.3f"%(test_auc))
    print("training AUC:%.3f"%(train_auc))
    
    print('best regularization parameter for cross-validation accuracy:%.2f'%(params[np.argmax(np.mean(np.array(test_scores),axis=0))]))
    print('best regularization parameter for AUC:%.2f'%(params[np.argmax(np.mean(np.array(test_prob_scores),axis=0))]))
    print('')

    return (test_scores,train_scores,test_prob_scores,train_prob_scores)


def getValidationFigure(filePath,title,xlab,ax,params,test_scores,train_scores,test_prob_scores,train_prob_scores):
    ax[0].semilogx(params,np.mean(np.array(test_scores),axis=0),marker='o',label='validation set')
    ax[0].semilogx(params,np.mean(np.array(train_scores),axis=0),marker='*',label='training set')

    ax[1].semilogx(params,np.mean(np.array(test_prob_scores),axis=0),marker='o',label='validation set')
    ax[1].semilogx(params,np.mean(np.array(train_prob_scores),axis=0),marker='*',label='training set')

    ax[0].legend(loc='best')
    ax[1].legend(loc='best')

    ax[0].set_xlabel(xlab,fontsize=16)
    ax[1].set_xlabel(xlab,fontsize=16)

    ax[0].set_ylabel('Accuracy',fontsize=18)
    ax[1].set_ylabel('AUC',fontsize=18)
    plt.suptitle(title,fontsize=20)
    return ax

def checkModel(clf,params,X_all,y,train_len,title,xlab,filePath):
    X_all_train=X_all[:train_len]   
    (test_scores,train_scores,test_prob_scores,train_prob_scores)=\
    getCrossValidation(X_all_train,y,clf,params)
    
    fig, ax = plt.subplots(1,2,figsize=(16.0, 10.0))

    ax=getValidationFigure(filePath,title,xlab,ax,params,test_scores,train_scores,test_prob_scores,train_prob_scores)
    plt.savefig(filePath)