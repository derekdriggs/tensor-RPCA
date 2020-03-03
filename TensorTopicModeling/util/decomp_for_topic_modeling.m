%%% Demo script for using RPCA for topic modeling. This script should
%   be run from RPCA_Topic_Model.cpp to decompose the whitened empirical
%   moment-tensor.
%
%   This script reads "matrix_for_rpca.txt", which contains an NxN^2
%   unwrapped moment-tensor, and decomposes this tensor into L+S, where
%   L is a low-rank tensor and S is sparse.
%
%   This script writes "phi.tx t", "lambda.txt", and "sparse_component.txt".
%   Phi is a matrix and lambda is a vector satisfying
%   L = sum_{i=1}^R lambda(i) [[phi(:,i), phi(:,i), phi(:,i)]].

X = load('../datasets/news/result_RPCA/matrix_for_rpca.txt');

M = size(X,1);
N = sqrt(size(X,2));
P = N;
R = M;

assert(M == N, 'Matricized tensor must be of dimensions M x M^2.')
X = reshape(X,M,N,P);

errFcn = @(G)norm(X(:)-G(:))/norm(X(:));

% symmetric initialisation for symmetric decomposition
A = randn(M,R);
B = A;
C = A;

% set parameters
params = struct('M',M,'N',N,'P',P,'R',R,'lambda',0.1,'mu',1e-8,'errFcn',errFcn,'A0',A,'B0',B,'C0',C);

% Set options for L-BFGS
params.progTol = 1e-8;
params.optTol  = 1e-8;
params.MaxIter = 1e3;
params.store   = 1;
params.gpu     = 0;

addpath(genpath('../../'))
[L,S,errHist] = tensor_RPCA(X,params);

Areco = L.A;

mults = ones(R,1);

dlmwrite('../datasets/news/result_RPCA/phi.txt',Areco,'delimiter','\t')
dlmwrite('../datasets/news/result_RPCA/lambda.txt',mults,'delimiter','\t')
dlmwrite('../datasets/news/result_RPCA/sparse_component.txt',S,'delimiter','\t')


exit;