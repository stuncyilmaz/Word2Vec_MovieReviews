__author__ = "Sandeep"

#####################################################
"""
    Code to generate word2vec based clusters and corresponding
    dictionary for transalting the original text.
    Using this dictionary, original text is translated by mapping
    all the words in the topic to a sigle word
"""
######################################################

import pandas as pd
import numpy as np
import nltk
import string
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import nltk.data
from nltk import stem
from gensim.models import word2vec
from sklearn.cluster import KMeans
import time
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import json


 def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    
    trnsfm_d = re.sub(r'\b(?:didnot|didn\'t|not|never|no|n\'t)\b[\w\s]+[^\w\s]',
                     lambda match: re.sub(r'(\s+)(\w+)', r'\1NEG\2', match.group(0)),
                     review_text, flags=re.IGNORECASE)
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", trnsfm_d)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.decode('utf-8').strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

def create_bag_of_centroids( wordlist, word_centroid_map, tfidf_scores ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += tfidf_scores[word]
    #
    # Return the "bag of centroids"
    return bag_of_centroids

def modify_text( wordlist, word_centroid_map ):
    
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    sentence = []
    for word in wordlist:
        if word in word_centroid_map:
            sentence.append(word_centroid2word_map[word])
        else:
            sentence.append( word)
    #
    # Return the "bag of centroids"
    return sentence


if __name__ == "__main__":


    # Read data from files 
    train = pd.read_csv( "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
    test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
    unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )

    # Verify the number of reviews that were read (100,000 in total)
    print "Read %d labeled train reviews, %d labeled test reviews, " \
     "and %d unlabeled reviews\n" % (train["review"].size,  
     test["review"].size, unlabeled_train["review"].size )

    # Load the punkt tokenizer
    tokenizer = nltk.data.load('/Users/sandeepvanga/nltk_data/tokenizers/punkt/english.pickle')
    print tokenizer

    sentences = []  # Initialize an empty list of sentences

    print "Parsing sentences from training set"
    for review in train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    print "Parsing sentences from unlabeled set"
    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    print len(sentences)

    print sentences[0]

    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality                      
    min_word_count = 40   # Minimum word count                        
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size                                                                                    
    downsampling = 1e-3   # Downsample setting for frequent words

    print num_features

    # Initialize and train the model (this will take some time)
    print "Training model..."
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)

    # If you don't plan to train the model any further, calling 
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and 
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)

    print model.syn0.shape

    model["flower"]

    start = time.time() # Start time

    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.syn0
    #num_clusters = word_vectors.shape[0] / 5
    num_clusters = 1000

    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans( n_clusters = num_clusters )
    idx = kmeans_clustering.fit_predict( word_vectors )

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print "Time taken for K Means clustering: ", elapsed, "seconds."

    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number                                                                                            
    word_centroid_map = dict(zip( model.index2word, idx ))

    word_centroid2word_map = {}
    for cluster in xrange(0,num_clusters):
        #
        # Print the cluster number  
        print "\nCluster %d" % cluster
        #
        # Find all of the words for that cluster number, and print them out
        words = []
        for i in xrange(0,len(word_centroid_map.values())):
            if( word_centroid_map.values()[i] == cluster ):
                words.append(word_centroid_map.keys()[i])
                
        random_word = random.choice(words)
        
        for word in words:
            word_centroid2word_map[word] = random_word
        #print words



    ## For the first 10 clusters
    #for cluster in xrange(0,10):
    #    #
    #    # Print the cluster number  
    #    print "\nCluster %d" % cluster
    #    #
    #    # Find all of the words for that cluster number, and print them out
    #    words = []
    #    for i in xrange(0,len(word_centroid_map.values())):
    #        if( word_centroid_map.values()[i] == cluster ):
    #            words.append(word_centroid_map.keys()[i])
    #    print words

    #print word_centroid2word_map


    # ****************************************************************
    # Translate the text based on topics obtained from clustering

    clean_train_reviews = []

    for review in train["review"]:
        clean_train_reviews.append( review_to_wordlist( review, \
            remove_stopwords=True ))

    print "Training data is cleaned"

    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append( review_to_wordlist( review, \
            remove_stopwords=True ))

    print "Testing data is cleaned"

    print "Parsing sentences from train set"
    modified_training_set = []
    for review in clean_train_reviews:
        modified_training_set.append(" ".join(modify_text(review, word_centroid_map)))
    train_set = {}
    train_set["review"] = modified_training_set

    print "Parsing sentences from test set"
    modified_testing_set = []
    for review in clean_test_reviews:
        modified_testing_set.append(" ".join(modify_text(review, word_centroid_map)))
    test_set = {}
    test_set["review"] = modified_testing_set
        
    outJSON=dict(train_set)
    outputFName='train_modified_1000.json'
    with open(outputFName, 'wb') as outfile:
        json.dump(outJSON, outfile)
        
    outJSON=dict(test_set)
    outputFName='test_modified_1000.json'
    with open(outputFName, 'wb') as outfile:
        json.dump(outJSON, outfile)
