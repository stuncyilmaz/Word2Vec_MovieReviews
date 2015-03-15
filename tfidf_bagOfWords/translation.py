
# coding: utf-8

# In[1]:

import gensim
import pandas as pd
import os
import cPickle


# In[2]:

import numpy as np
import sys
import nltk
import re
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from bs4 import BeautifulSoup 
from  sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt


from nltk.probability import FreqDist
from itertools import chain
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# #Paths

# In[3]:

original_data_path='../original_data/'
saved_data_path='../saved_data/'
plots_path='../plots/'

codes_path='../codes/'
if not codes_path in sys.path: sys.path.append(codes_path)
import chooseFeature
import classification_functions


# In[4]:


inputFName=saved_data_path+'modified_text_with_3500_clusters/train_modified_3500.json'
inJSON = json.load(open(inputFName, "r"))
train=pd.DataFrame(inJSON)
inJSON =None


inputFName=saved_data_path+'modified_text_with_3500_clusters/test_modified_3500.json'
inJSON = json.load(open(inputFName, "r"))
test=pd.DataFrame(inJSON)
inJSON =None

train['review_cleaned']=train['review'].map(lambda x: " ".join(x))
test['review_cleaned']=test['review'].map(lambda x: " ".join(x))


# In[5]:

train_original=pd.read_csv(original_data_path+'labeledTrainData.tsv', header=0, delimiter="\t", quoting=3 )
train['sentiment']=train_original['sentiment']
train_original=None
train_test=pd.DataFrame(pd.concat([train['review_cleaned'],test['review_cleaned']]))


# # check bigram and unigram numbers

# In[6]:

all_words_train=train['review_cleaned'].map(lambda x:x.split())
all_words_test=test['review_cleaned'].map(lambda x:x.split())
all_words=all_words_test+all_words_train

bigrams_all=[]
for elt in all_words:
    bigrams_all+=list(nltk.bigrams(elt))
    
bigrams_all_fd = FreqDist(bigrams_all)

all_words_train=list(chain.from_iterable(all_words_train))
all_words_test=list(chain.from_iterable(all_words_test))

all_words=all_words_train+all_words_test
all_words_test,all_words_train=None,None

all_words_fd = FreqDist(all_words)


# In[7]:

print('all_words > 0:%i,bigrams_all>0: %i '%(len(set(all_words)),len(set(bigrams_all))))


# In[8]:

bigrams_cut_off_2=list(set([k for k, v in bigrams_all_fd.iteritems()  if v>2]))
unigrams_cut_off_2=list(set([k for k, v in all_words_fd.iteritems()  if v>2]))
print('unigrams_cut_off_2:%i,bigrams_cut_off_2: %i '%(len(set(unigrams_cut_off_2)),len(set(bigrams_cut_off_2))))


# In[9]:

bigrams_cut_off_5=list(set([k for k, v in bigrams_all_fd.iteritems()  if v>5]))
unigrams_cut_off_5=list(set([k for k, v in all_words_fd.iteritems()  if v>5]))
print('unigrams_cut_off_5:%i,bigrams_cut_off_5: %i '%(len(set(unigrams_cut_off_5)),len(set(bigrams_cut_off_5))))


# In[10]:

all_words,bigrams_all,unigrams_cut_off_2,bigrams_cut_off_2,unigrams_cut_off_5,bigrams_cut_off_5=None,None,None,None,None,None


# # create tfidf models

# In[11]:

tfidfer_bigrams_cut_off_2 = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words =None)

tfidfer_bigrams_cut_off_2.fit(list(train_test['review_cleaned']))
X_bigrams_cut_off_2=tfidfer_bigrams_cut_off_2.transform(train_test['review_cleaned'])
tfidfer_bigrams_cut_off_2=None


# In[12]:

tfidfer_bigrams_cut_off_5 = TfidfVectorizer(min_df=6,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words =None)


tfidfer_bigrams_cut_off_5.fit(list(train_test['review_cleaned']))
X_bigrams_cut_off_5=tfidfer_bigrams_cut_off_5.transform(train_test['review_cleaned'])
tfidfer_bigrams_cut_off_5=None


# In[13]:

tfidfer_bigrams_cut_off_10 = TfidfVectorizer(min_df=11,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words =None)


tfidfer_bigrams_cut_off_10.fit(list(train_test['review_cleaned']))
X_bigrams_cut_off_10=tfidfer_bigrams_cut_off_10.transform(train_test['review_cleaned'])
tfidfer_bigrams_cut_off_10=None


# In[14]:

tfidfer_unigrams_cut_off_2 = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 1), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words =None)

tfidfer_unigrams_cut_off_2.fit(list(train_test['review_cleaned']))
X_unigrams_cut_off_2=tfidfer_unigrams_cut_off_2.transform(train_test['review_cleaned'])
tfidfer_unigrams_cut_off_2=None


# In[15]:

tfidfer_trigrams_cut_off_2 = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words =None)

tfidfer_trigrams_cut_off_2.fit(list(train_test['review_cleaned']))
X_trigrams_cut_off_2=tfidfer_trigrams_cut_off_2.transform(train_test['review_cleaned'])
tfidfer_trigrams_cut_off_2=None


# In[16]:

tfidfer_trigrams_cut_off_10 = TfidfVectorizer(min_df=11,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words =None)

tfidfer_trigrams_cut_off_10.fit(list(train_test['review_cleaned']))
X_trigrams_cut_off_10=tfidfer_trigrams_cut_off_10.transform(train_test['review_cleaned'])
tfidfer_trigrams_cut_off_10=None


# In[17]:

tfidfer_bigrams_cut_off_40 = TfidfVectorizer(min_df=41,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words =None)


tfidfer_bigrams_cut_off_40.fit(list(train_test['review_cleaned']))
X_bigrams_cut_off_40=tfidfer_bigrams_cut_off_40.transform(train_test['review_cleaned'])
tfidfer_bigrams_cut_off_40=None


# In[18]:

tfidfer_bigrams_cut_off_200 = TfidfVectorizer(min_df=201,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words =None)


tfidfer_bigrams_cut_off_200.fit(list(train_test['review_cleaned']))
X_bigrams_cut_off_200=tfidfer_bigrams_cut_off_200.transform(train_test['review_cleaned'])


# In[19]:

X_unigrams_cut_off_2.shape[1],X_bigrams_cut_off_5.shape[1],X_bigrams_cut_off_2.shape[1],X_bigrams_cut_off_10.shape[1],X_bigrams_cut_off_40.shape[1],X_bigrams_cut_off_200.shape[1],X_trigrams_cut_off_2.shape[1],X_trigrams_cut_off_10.shape[1]



# # Cross validations

# In[20]:

y=train['sentiment']
train_len=train.shape[0]

clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, 
                         C=1, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight=None, random_state=None)
params=[0.1,0.5,1,2,3,4,5,6,7,14,18,20,21,22,23,30,40,50,75,100,500]



##  bigrams_cut_off_5
print('bigrams_cut_off_5\n')
X_all=X_bigrams_cut_off_5
filePath=plots_path+'logistic_bigrams_cut_off_5_translated3500Clusters.png'
title='Logistic Regression'
title+=' 5-fold cross-validation'
title+='\nbigrams with frequency cut-off 5'
xlab='regularization (logistic regression C parameter)'
classification_functions.checkModel(clf,params,X_all,y,train_len,title,xlab,filePath)

##  trigrams_cut_off_2
print('trigrams_cut_off_2\n')
X_all=X_trigrams_cut_off_2
filePath=plots_path+'logistic_trigrams_cut_off_2_translated3500Clusters.png'
title='Logistic Regression'
title+=' 5-fold cross-validation'
title+='\ntrigrams with frequency cut-off 2'
xlab='regularization (logistic regression C parameter)'
classification_functions.checkModel(clf,params,X_all,y,train_len,title,xlab,filePath)

##  bigrams_cut_off_2
print('bigrams_cut_off_2\n')
X_all=X_bigrams_cut_off_2
filePath=plots_path+'logistic_bigrams_cut_off_2_translated3500Clusters.png'
title='Logistic Regression'
title+=' 5-fold cross-validation'
title+='\nbigrams with frequency cut-off 2'
xlab='regularization (logistic regression C parameter)'
classification_functions.checkModel(clf,params,X_all,y,train_len,title,xlab,filePath)


##  unigrams_cut_off_2
print('unigrams_cut_off_2\n')
X_all=X_unigrams_cut_off_2
filePath=plots_path+'logistic_unigrams_cut_off_2_translated3500Clusters.png'
title='Logistic Regression'
title+=' 5-fold cross-validation'
title+='\nunigrams with frequency cut-off 2'
xlab='regularization (logistic regression C parameter)'
classification_functions.checkModel(clf,params,X_all,y,train_len,title,xlab,filePath)


##  bigrams_cut_off_10
print('bigrams_cut_off_10\n')
X_all=X_bigrams_cut_off_10
filePath=plots_path+'logistic_bigrams_cut_off_10_translated3500Clusters.png'
title='Logistic Regression'
title+=' 5-fold cross-validation'
title+='\nbigrams with frequency cut-off 10'
xlab='regularization (logistic regression C parameter)'
classification_functions.checkModel(clf,params,X_all,y,train_len,title,xlab,filePath)

##  bigrams_cut_off_40
print('bigrams_cut_off_40\n')
X_all=X_bigrams_cut_off_40
filePath=plots_path+'logistic_bigrams_cut_off_40_translated3500Clusters.png'
title='Logistic Regression'
title+=' 5-fold cross-validation'
title+='\nbigrams with frequency cut-off 40'
xlab='regularization (logistic regression C parameter)'
classification_functions.checkModel(clf,params,X_all,y,train_len,title,xlab,filePath)


##  bigrams_cut_off_200
print('bigrams_cut_off_200\n')
X_all=X_bigrams_cut_off_200
filePath=plots_path+'logistic_bigrams_cut_off_200_translated3500Clusters.png'
title='Logistic Regression'
title+=' 5-fold cross-validation'
title+='\nbigrams with frequency cut-off 200'
xlab='regularization (logistic regression C parameter)'
classification_functions.checkModel(clf,params,X_all,y,train_len,title,xlab,filePath)



##  trigrams_cut_off_10
print('trigrams_cut_off_10\n')
X_all=X_trigrams_cut_off_10
filePath=plots_path+'logistic_trigrams_cut_off_10_translated3500Clusters.png'
title='Logistic Regression'
title+=' 5-fold cross-validation'
title+='\ntrigrams with frequency cut-off 10'
xlab='regularization (logistic regression C parameter)'
classification_functions.checkModel(clf,params,X_all,y,train_len,title,xlab,filePath)


# # create kaggle submission

# In[21]:

clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, 
                         C=1, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight=None, random_state=None)

clf.set_params(C=100)
X_all=X_bigrams_cut_off_2
X_all_train=X_all[:train_len]  
X_all_test=X_all[train_len:]  


clf.fit(X_all_train,y)

predicted=clf.predict_proba(X_all_test)[:,1]

inputFName=saved_data_path+'testSet.pkl'
with open(inputFName,'rb') as fp:
    test_current=cPickle.load(fp)

test_current['sentiment']=predicted
test_current['id']=test_current['id'].map(lambda x:x.replace('"', '').strip())
test_current=test_current[['id','sentiment']]
test_current.to_csv(saved_data_path+'logisticRegression_bigrams_cut_off_2__translated3500Clusters_submission.csv',index=False)


# In[ ]:



