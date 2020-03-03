function [Lcomp,S,errHist] = tensor_RPCA(X,params)
% [L,S,errHist] = solver_split_SPCP(Y,params)
% Finds a stationary point to approximately solve the problem
%
%   minimize_{A,B,C} 1/2*|| [[A,B,C]] + S - X ||^2 + lambda*||S||_1
%
%                   + mu/2*sum_{r=1}^R ||a_r||^3 + ||b_r||^3 + ||c_r||^3,
%
% where A is MxR, B is NxR, C is PxR, and S and X are MxNxP.
%
% Outputs:
%   Lcomp: Low-rank tensor and components satisfying L=[[A,B,C]]
%   S: Sparse tensor
%   errHist:
%       errHist(:,1) is a record of runtime
%       errHist(:,2) is a record of the full objective (.f*resid^2 + lambda_L,
%           etc.)
%       errHist(:,3) is the output of params.errFcn if provided
%
% Input: params is a structure with optional fields
%   errFcn     to compute objective or error (empty)
%   R          desired rank of L (10)
%   A0,B0,C0   initial points for L0 = [[A0,B0,C0]] (random)
%   gpu        1 for gpu, 0 for cpu (0)
%   lambda     l1 penalty weight (0.8)
%   mu         nuclear norm penalty weight (115)
%   plot_err   plot the error history upon exit (0)
%   
tic;

% in case X is a tensor
X = double(X);

[M, N, P]   = size(X);
params.M = M;
params.N = N;
params.P = P;

errFcn = setOpts(params,'errFcn',[]);
R      = setOpts(params,'R',10);
A0     = setOpts(params,'A0',randn(M,R));
B0     = setOpts(params,'B0',randn(N,R));
C0     = setOpts(params,'C0',randn(P,R));
gpu    = setOpts(params,'gpu',0);
lambda = setOpts(params,'lambda',0.8);
mu     = setOpts(params,'mu',115);
plot_err = setOpts(params,'plot_err',0);

% check if we are on the GPU
if strcmp(class(X), 'gpuArray')
	gpu = 1;
end

if gpu
    A0 = gpuArray(A0);
    B0 = gpuArray(B0);
    C0 = gpuArray(C0);
end

% initial point
x0 = [vec(A0); vec(B0); vec(C0)];

% set necessary parameters
params.lambda = lambda;
params.mu     = mu;
params.gpu    = gpu;

% objective/gradient map for L-BFGS solver
ObjFunc = @(x)func_tensor_rpca(x,X,params,errFcn);

try
    func_tensor_rpca();
catch
    error('func_tensor_rpca is undefined. Did you run setup_tensor_RPCA.m?')
end

% solve using L-BFGS
[x,~,~] = lbfgs_gpu(ObjFunc,x0,params);

errHist=func_tensor_rpca();
if ~isempty( errHist ) && plot_err
    figure;
    semilogy( errHist );
end

% prepare output
A = reshape(x(1:M*R),M,R);
B = reshape(x(M*R+1:R*(M+N)),N,R);
C = reshape(x(R*(M+N)+1:R*(M+N+P)),P,R);
S = func_tensor_rpca(x,X,params,'S');
S = reshape(S,M,N,P);

L = full(ktensor({A,B,C}));

Lcomp = struct('A',A,'B',B,'C',C,'L',L);

end



function out = setOpts(options, opt, default)
    if isfield(options, opt)
        out = options.(opt);
    else
        out = default;
    end
end