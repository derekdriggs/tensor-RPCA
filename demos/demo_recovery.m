%% Recreate experiment from the paper

%% set parameters
params.lambda  = 1e-3; % set to 1e-3 to match paper
params.mu      = 1e-5; % set to 1e-5 to match paper
params.MaxIter = 1e3;  % set to 1e3 to match paper
params.M       = 20;
params.N       = 20;
params.P       = 20;

M = 20;
N = 20;
P = 20;

% initialise using CP-decomposition? (Not recommended, but useful for testing.)
% otherwise use random initial points.
cp_init = 0;

Rmax   = 60;
rhomax = 20;

% store errors in matrix
ErrMat = zeros(Rmax,rhomax);

%% Run tests
for rnk = 20:Rmax
    
    R = rnk+10;
    params.R = R;
    
    for rhoS = 10:rhomax
        
        % Make a rank-rnk random tensor
        U1 = randn(20,rnk);
        U2 = randn(20,rnk);
        U3 = randn(20,rnk);
        L = full( ktensor( {U1,U2,U3} ) );
        
        % measure error with respect to true low-rank component
        params.errFcn  = @(A)norm(L(:)-A(:))/norm(L(:));
        
        S = zeros(20,20,20);
        
        % Make a sparsity-rhoS/100 random sparse tensor
        for k = 1:20
            S(:,:,k) = sprandn(20,20,rhoS/100);
        end
        
        Z = L + S;
        
        if cp_init
            % Generate initial points from CP-decomposition
            CP1 = cp_als( Z, 20 );
            A = CP1.U{1};
            B = CP1.U{2};
            C = CP1.U{3};
        else
            % Generate random initial points
            A = randn(20,rnk + 10);
            B = randn(20,rnk + 10);
            C = randn(20,rnk + 10);
        end
        
        [Lreco,Sreco,errHist] = tensor_RPCA(Z,params);
        
        ErrMat(rnk,rhoS) = min(errHist(:,3));
    end
end

