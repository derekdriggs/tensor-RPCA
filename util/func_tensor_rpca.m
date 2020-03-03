function [f, df] = func_tensor_rpca(x,Z,params,errFcn)
% [f, df] = func_tensor_rpca(x,Y,params,errFunc)
% [errHist] = func_split_spcp();
% [S] = func_split_spcp(x,Y,params,'S');
%
% Compute function and gradient of split-SPCP objective
%
%   1/2*|| [[A,B,C]] + S - Z ||^2 + lambda*||S||_1
%
%           + mu*sum_{r=1}^R ||a_r||^3 + ||b_r||^3 + ||c_r||^3
%
% where A is MxR, B is NxR, C is PxR, and S and Z are MxNxP.
%
% See also: func_sym_tensor_rpca.m (use when decomposing symmetric tensors).

% initialise error history if that is all we want
persistent errHist
if nargin==0
   f = errHist;
   errHist = [];
   return;
end

if nargin<3, params=[]; end

mu     = params.mu;
lambda = params.lambda;
M      = params.M;
N      = params.N;
P      = params.P;
R      = params.R;
useGPU = params.gpu;

A = reshape(x(1:M*R),M,R);
B = reshape(x(M*R+1:R*(M+N)),N,R);
C = reshape(x(R*(M+N)+1:R*(M+N+P)),P,R);

L = full(ktensor({A,B,C}));
S = reshape(sign(Z(:)-L(:)).*max(abs(Z(:)-L(:)) - lambda,0),M,N,P);

% return S if that is all we want
if nargout==1
    f = S;
    return;
end

tic;
for dim = 1:3
    if dim == 1
        Al = khatrirao(C,B);
        b = Z-S;
        b = tenmat(b, 1, 'fc');
        b = double(b.data)';
        A = A';

        for i = 1:size(b,2)
            Agrad(:,i) = Al'*(Al*A(:,i) - b(:,i)) + 3*mu*norm(A(:,i)).*A(:,i);
        end

        Agrad = Agrad';
        A = A';

    elseif dim == 2
        Al = khatrirao(A,C);
        b = Z-S;
        b = tenmat(b, 2, 'fc');
        b = double(b.data)';
        B = B';

        for i = 1:size(b,2)
            Bgrad(:,i) = Al'*(Al*B(:,i) - b(:,i)) + 3*mu*norm(B(:,i)).*B(:,i);
        end

        Bgrad = Bgrad';
        B = B';

    else
        Al = khatrirao(B,A);
        b = Z-S;
        b = tenmat(b, 3, 'fc');
        b = double(b.data)';
        C = C';
        for i = 1:size(b,2)
            Cgrad(:,i) = Al'*(Al*C(:,i) - b(:,i)) + 3*mu*norm(C(:,i)).*C(:,i);
        end

        Cgrad = Cgrad';
        C = C';

    end 
end

if useGPU
    df = gpuArray( [Agrad(:); Bgrad(:); Cgrad(:)] );
else
    df = [Agrad(:); Bgrad(:); Cgrad(:)];
end

f = 0.5*norm(L(:)+S(:)-Z(:))^2 + lambda*norm(S(:),1);
for i = 1:size(A,2)
    f = f + mu*(norm(A(:,i))^3 + norm(B(:,i))^3 + norm(C(:,i))^3);
end

errHist(end+1,1) = toc;
errHist(end,2) = gather(f);

if ~isempty(errFcn)
    errHist(end,3) = errFcn(L);
end

tic;

end