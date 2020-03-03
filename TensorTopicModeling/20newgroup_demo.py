# Python script to compare topic modeling using non-negative matrix
# factorization, LDA fit with Gibbs Sampling, LDA fit using alternating least
# squares, and LDA fit using Tensor RPCA.
#
# This script recreates the topic modeling experiments in "Tensor RPCA: Better
# recovery through atomic norm regularisation."
#
# Dependencies: TopicModelingSingleNodeALS by Furong Huang
#
# Written by Derek Driggs, 2019.

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import numpy as np

import os

n_train_docs = 10000
n_test_docs  = 100
n_words      = 100
n_topics     = 5
cats         = ['talk.religion.misc','comp.graphics','sci.space','rec.sport.hockey','rec.autos']
n_top_words  = 10

# prints the top words associated with the topics learned by 'model'
def print_top_words_model(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx+1))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    print()

# prints the top words associated with a given matrix of topic-word distributions
def print_top_words_dist(word_topic_dist, feature_names, n_topics, n_top_words):
    for j in range(n_topics):
        print("Topic %d:" % (j+1))
        topic = word_topic_dist[:,j]
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


# Load the 20 newsgroups dataset. We remove headers, footers, and quotes, as
# well as common English stop-words, and words occurring only in one document
# or in more than 95% of the documents.
print("Loading dataset...")
t0 = time()
dataset = fetch_20newsgroups(shuffle=True, random_state=1,categories=cats,
                             remove=('headers', 'footers', 'quotes'), subset='train')

dataset_test = fetch_20newsgroups(shuffle=True, random_state=1,categories=cats,
                             remove=('headers', 'footers', 'quotes'), subset='test')

data_train = dataset.data[:n_train_docs]
data_test  = dataset_test.data[:n_test_docs]

# If we ask for more documents than are available
n_train_docs = len(data_train)
n_test_docs = len(data_test)

print("done in %0.3fs." % (time() - t0))

# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_words,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_train)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_words,
                                stop_words='english')

tf_test_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_words,
                                stop_words='english')

t0 = time()
tf = tf_vectorizer.fit_transform(data_train)
tf_test = tf_test_vectorizer.fit_transform(data_test)
print("done in %0.3fs." % (time() - t0))

# Fit the NMF model
print("Fitting the NMF model with tf-idf features, "
      "n_train_docs=%d and n_words=%d..."
      % (n_train_docs, n_words))
t0 = time()
nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5)

nmf.fit(tfidf)

print("done in %0.3fs." % (time() - t0))

# Fit the LDA model
print("Fitting LDA models with tf features, "
      "n_train_docs=%d and n_words=%d..."
      % (n_train_docs, n_words))
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=1)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

# compute perplexity
lda_perplexity = lda.perplexity(tf_test)

print("\nRunning Tensor ALS Model....")
print("\nWriting training data...")

if ~os.path.isfile('datasets/news/news_train.txt'):
    # Write training data
    fh = open("datasets/news/news_train.txt", "w")
    for i in range(n_train_docs):
        for j in range(0,n_words):
            fh.write( str(i + 1) + "       " + str(j + 1) + "       " + str( tf[i,j] ) + "\n" )
    fh.close()
else:
    print("\nTraining data already exists")

print("\nFinished writing training data")
print("\ndone in %0.3fs." % (time() - t0))
print("\nWriting test data")

if ~os.path.isfile("datasets/news/news_test.txt"):
    # Write test data
    fh2 = open("datasets/news/news_test.txt", "w")
    for i in range(n_test_docs):
        for j in range(0,n_words):
            fh2.write( str(i + 1) + "       " + str(j + 1) + "       " + str( tf_test[i,j] ) + "\n" )
    fh2.close()
else:
    print("\nTest data already exists")

print("\nFinished writing test data")
print("\ndone in %0.3fs." % (time() - t0))

# Define necessary directories
data_dir = 'datasets/news/'
als_res_dir  = 'datasets/news/result_ALS/'
rpca_res_dir = 'datasets/news/result_RPCA/'
als_exe_dir  = 'dependency/TensorDecomposition4TopicModeling/TopicModelingSingleNodeALS/TopicModel/TopicModel/'

als_str_exe = als_exe_dir + './exe-topicmodel ' + str(n_train_docs) + ' ' + str(n_test_docs) + ' ' + str(n_words)\
+ ' ' + str(n_topics) + ' 0.01 1 ' + data_dir + 'news_train.txt '\
+ data_dir + 'news_test.txt ' + als_res_dir + 'news_corpus_topic_weights.txt '\
+ als_res_dir + 'news_topic_word_matrix.txt ' + als_res_dir + 'news_inferred_topic_weights_per_document.txt'

rpca_str_exe = './' + 'exe-rpca_topicmodel ' + str(n_train_docs) + ' ' + str(n_test_docs) + ' ' + str(n_words)\
+ ' ' + str(n_topics) + ' 0.01 1 ' + data_dir + 'news_train.txt '\
+ data_dir + 'news_test.txt ' + rpca_res_dir + 'rpca_news_corpus_topic_weights.txt '\
+ rpca_res_dir + 'rpca_news_topic_word_matrix.txt ' + rpca_res_dir + 'rpca_news_inferred_topic_weights_per_document.txt'

os.system(als_str_exe)
os.system(rpca_str_exe)

# Print top words for each topic
print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words_model(nmf, tfidf_feature_names, n_top_words)

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words_model(lda, tf_feature_names, n_top_words)

print("\nTopics in ALS tensor model:")
x_als = np.loadtxt("datasets/news/result_ALS/news_topic_word_matrix.txt")
print_top_words_dist(x_als, tf_feature_names, n_topics, n_top_words)

print("\nTopics in RPCA tensor model:")
x_rpca = np.loadtxt("datasets/news/result_RPCA/rpca_news_topic_word_matrix.txt")
print_top_words_dist(x_rpca, tf_feature_names, n_topics, n_top_words)
