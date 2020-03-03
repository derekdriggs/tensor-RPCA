LIB_DIR=dependency/TensorDecomposition4TopicModeling/TopicModelingSingleNodeALS/TopicModel/TopicModel
SRC=$(LIB_DIR)/Probability.cpp $(LIB_DIR)/Pvalue.cpp $(LIB_DIR)/Spectral.cpp $(LIB_DIR)/Util.cpp
CFLAGS=-lsvd -lm
COPTZ=-O3 -g  -I.
SUPP_WARN=-Wno-write-strings
N_train = 10000
N_test = 100
Voca_size=100
Hidden_size=3
ALPHA0=0.01
IndexStart=1
DataFolder =../datasets/synthetic
Sample_train=samples_train.txt
Sample_test = samples_test.txt
ResultFolder= $(DataFolder)/result
Result_topic_eigenvalue = corpus_topic_weights.txt
Result_topic_eigenvector = topic_word_matrix.txt
Result_topic_inferred_membership = inferred_topic_weights_per_document.txt

Tensor_dimension = 3
Tensor_file=Tensor.txt
Result_tensor_eigenvalue = tensor_eigenvalue.txt
Result_tensor_eigenvector=tensor_eigenvector.txt

exe-rpca_topicmodel: RPCA_Topic_Model.cpp util/IOdd.cpp $(LIB_DIR)/Probability.cpp $(LIB_DIR)/Pvalue.cpp $(LIB_DIR)/Spectral.cpp $(LIB_DIR)/stdafx.cpp util/stdafx_dd.cpp $(LIB_DIR)/Util.cpp
	g++ -o $@ $^ $(COPTZ) $(SUPP_WARN) -DNDEBUG -std=c++0x

