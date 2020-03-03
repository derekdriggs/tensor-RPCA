// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#ifndef __TopicModel__stdafx__
#define __TopicModel__stdafx__
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <list>
#include <vector>
#include <set>
#include <queue>
#include <map>

#include <iterator>

#include <math.h>
#include <algorithm>

#include "dependency/TensorDecomposition4TopicModeling/TopicModelingSingleNodeALS/dependency/Eigen/Dense"
#include "dependency/TensorDecomposition4TopicModeling/TopicModelingSingleNodeALS/dependency/Eigen/Sparse"
#include "dependency/TensorDecomposition4TopicModeling/TopicModelingSingleNodeALS/dependency/Eigen/Core"
#include "dependency/TensorDecomposition4TopicModeling/TopicModelingSingleNodeALS/dependency/Eigen/OrderingMethods"
#include "dependency/TensorDecomposition4TopicModeling/TopicModelingSingleNodeALS/dependency/Eigen/SparseQR"

#include <ctime>
#include <sys/time.h>   // this is for linux machine
//#include <windows.h> // this is for windows machine



//reference additional headers your program requires here
#include "IOdd.h"
#include "../dependency/TensorDecomposition4TopicModeling/TopicModelingSingleNodeALS/TopicModel/TopicModel/Util.h"

#include "../dependency/TensorDecomposition4TopicModeling/TopicModelingSingleNodeALS/TopicModel/TopicModel/Probability.h"
#include "../dependency/TensorDecomposition4TopicModeling/TopicModelingSingleNodeALS/TopicModel/TopicModel/Spectral.h"
#include "../dependency/TensorDecomposition4TopicModeling/TopicModelingSingleNodeALS/TopicModel/TopicModel/Pvalue.h"



#define LEARNRATE 1e-9
#define MINITER 100
#define MAXITER 500
#define TOLERANCE 1e-6
#define NORMALIZE
#define ORTH_COST_WEIGHT 1000
#endif




//#define LEARNRATE 1e-9 // learning rate for tensor decomposition
//#define MINITER 10000 // minimum number of iterations
//#define MAXITER 100000 // maximum number of iterations
//#define PVALUE_TOLE 0.01 // pvalue tolerance
// #define KTRUE 6003//159//6003
