%% Demo of tensor RPCA on a video clip

% Load the data:
downloadEscalatorData;
load escalator_data % contains X (data), m and n (height and width)
X = double(X);
%%

run('~/GitHub/fastRPCA/setup_fastRPCA.m')

load 'escalator_data';
X = double(X);

M = m;
N = n;
P = 200;

R = 50;

Y = zeros(m,n,200);

for k = 1:200
    Y(:,:,k) = reshape(X(:,k),m,n);
end

X = Y;

clear Y

A = randn(m,R);
B = randn(n,R);
C = randn(200,R);

%45 and 10 are the best so far
params        = struct('M',m,'N',n,'P',200,'R',size(A,2),'lambda',30,'mu',0.1);
params.x0     = [A(:); B(:); C(:)];
params.progTol = 1e-15;
params.optTol  = 1e-5;
params.MaxIter = 200;
params.store    = 1;

[Lreco,Sreco,errHist] = tensor_RPCA(X,params);

Areco = Lreco.A;
Breco = Lreco.B;
Creco = Lreco.C;

Lreco = full(ktensor({Areco,Breco,Creco}));


%% show all together in movie format

mat  = @(x) reshape( x, m, n );
figure(1); clf;
colormap( 'Gray' );

for k = 1:200
    imagesc( [X(:,:,k), double( Lreco(:,:,k) ), double( Sreco(:,:,k) )] );
    axis off
    axis image
    drawnow;
    pause(.05);  
end
