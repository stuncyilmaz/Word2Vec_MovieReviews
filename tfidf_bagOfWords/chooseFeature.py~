import collections
import sklearn
import pandas as pd
import os
import numpy as np 
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import datasets
from pprint import pprint

class chooseFeature(sklearn.base.BaseEstimator):
    """
    This defines a classifier that predicts on the basis of
      the feature that was found to have the best weighted purity, based on splitting all
      features according to their mean value. Then, for that feature split, it predicts
      a new example based on the mean value of the chosen feature, and the majority class for
      that split.
      You can of course define more variables!
    """

    def __init__(self):
        # if we haven't been trained, always return 1
        self.classForGreater= 1
        self.classForLeq = 1
        self.chosenFeature = 0
        self.type = "chooseFeatureClf"
        self.mean=None

    def impurity(self, labels):
        '''
         input: a numpy array of discrete categorical variables as the class labels
         output: Gini impurity of the labels

        '''

        if labels.shape[0]==0:
            return 0
        else:
            class_prob=[list(labels).count(i)/float(labels.shape[0]) for i in np.unique(labels)]
            entropy=sum([-i*np.log(i) for i in class_prob ])
            return entropy

    def weighted_impurity(self, list_of_label_lists):
        '''
         input: a list of numpy arrays for a split for a particular variable
         output: weighted entropy of the labels
        '''

        ginis=[self.impurity(label) for label in list_of_label_lists]
        weights=[label.shape[0] for label in list_of_label_lists]
        if np.sum(weights)==0: 
            return 0
        else:
            weighted_sum=sum([a*b for a,b in zip(ginis,weights)])/float(sum(weights))
            return weighted_sum

    def ftr_seln(self, data, labels):
        """return: index of feature with best weighted_impurity, when split
        according to its mean value; you are permitted to return other values as well,
        as long as the the first value is the index
        """
        gini_array=[]
        #loop over features
        for i in range(data.shape[1]):

            myMean=np.mean(data[:,i])

            classForGreater=labels[data[:,i]>myMean]

            classForLeq=labels[data[:,i]<=myMean]
            


            labels_list=[classForLeq,classForGreater]
            gini_array.append((self.weighted_impurity(labels_list)))
        
        self.chosenFeature=gini_array.index(min(gini_array))
            
        return self.chosenFeature


        # ## TODO: Your code here, uses weighted_impurity

    def fit(self, data, labels):
        """
        Inputs: data: a list of X vectors
        labels: Y, a list of target values
        """


        self.ftr_seln(data, labels)
        myMean=np.mean(data[:,self.chosenFeature])
        classForGreater=labels[data[:,self.chosenFeature]>myMean]
        classForLeq=labels[data[:,self.chosenFeature]<=myMean]

        myCounter=collections.Counter(classForLeq)
        self.classForLeq=myCounter.most_common()[0][0]

        if not np.array(classForGreater).size==0:

            myCounter=collections.Counter(classForGreater)
            self.classForGreater=myCounter.most_common()[0][0]
        #if classForGreater is empty, data uniform
        else: self.classForGreater=self.classForLeq
        
        self.mean=myMean
        
    def score(self,data, labels):
        return np.sum(self.predict(data)==labels)/np.float(labels.shape[0])


    def predict(self, testData):
        """
        Input: testData: a list of X vectors to label.
        Check the chosen feature of each
        element of testData and make a classification decision based on it
        """

        return [self.classForGreater if elt[self.chosenFeature]>self.mean else  self.classForLeq for elt in testData ]

def transform_sklearn_dictionary(input_dict):
    """ Input: input_dict: a Python dictionary or dictionary-like object containing
    at least information to populate a labeled dataset, L={X,y}
    return:
    X: a list of lists. The length of inner lists should be the number of features,
     and the length of the outer list should be the number of examples.
    y: a list of target variables, whose length is the number of examples.
    X & y are not required to be numpy arrays, but you may find it convenient to make them so.
    """
    X=np.asarray(input_dict['data'])
    y=np.asarray(input_dict['target'])
    return X, y


def transform_csv(data, target_col=0, ignore_cols=None):
    """ Input: data: a pandas DataFrame
    return: a Python dictionary with same keys as those used in sklearn's iris dataset
    (you don't have to create an object of the same data type as those in sklearn's datasets)
    """
    if target_col==0:target_col="target"
    if ignore_cols==None: ignore_cols=[]

    df_feature_names=[i for i in data.columns.values if not (i==target_col or i in ignore_cols)]
    df_data=data[df_feature_names]
    df_target=data[target_col]
    df_target_names=df_target.unique()

    my_dictionary={}
    my_dictionary['feature_names']=np.asarray(df_feature_names)
    my_dictionary['data']=np.asarray(df_data)
    my_dictionary["target"]=np.asarray(df_target)
    my_dictionary['target_names']=np.asarray(df_target_names)
    my_dictionary['DESCR']=""
    return my_dictionary

if __name__ == "__main__":
    rawdata=pd.read_csv('nursery.csv',header=0)
    nursery_dictionary= transform_csv(rawdata, target_col='target', ignore_cols=[])
    (X, y)=transform_sklearn_dictionary(nursery_dictionary)

    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    from sklearn.metrics import accuracy_score



    #Fill in arguments below for different value of k than default

    my_clf=chooseFeature()
    my_clf.fit(X,y)


    kf = cross_validation.KFold(len(X), n_folds=5)
    test_scores=[]
    train_scores=[]
    for train_idx, test_idx in kf:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]


        my_clf.fit(X_train, y_train)

        y_pred =my_clf.predict(X_test)
        y_true = y_test
        test_scores.append(accuracy_score(y_true, y_pred))

        y_pred =my_clf.predict(X_train)
        y_true = y_train
        train_scores.append(accuracy_score(y_true, y_pred))

    pprint(["train=%.3f, validation=%.3f"%(i,j) for i,j in np.dstack((train_scores,test_scores))[0]])
    pprint(["mean train accuracy =%.4f, mean validation accuracy=%.4f"%(np.mean(train_scores),np.mean(test_scores))])
    
