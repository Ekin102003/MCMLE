function [F,Y,p_iter,t]=MCMLE(L,D, c,lambda2,V,lambda1,n,gnd)
% Multiview Clustering via  Multiple  Laplacian Embeddings

% L£ºLaplacian matrix of all the view
% D£ºDegree matrix of all the view
% c£ºnumber of clusters
% V£ºnumber of views
% n£ºnumber of samples
% Y; indicator matrix

% We run Matlab2020b on a machine with an Intel Core i9-10940X CPU. 
% The ACC on BBC is 93.28%. 
% We found that the same experimental settings may obtain different results on different versions of MATLAB or different CPUs. 
% For example, under the same setting, we run Matlab2019b on a machine with an Intel Core i9-8700K CPU. 
% The ACC on BBC is 83.65%.

opts.record = 0;
opts.mxitr  = 100;%1000
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;
p_iter=zeros(1,100);
%% ------------------- 1. initialize ------------------
w=zeros(1,V);
for i=1:V
    w(1,i)=1/V;
end
%% ------------------- 2. first iteration ------------------
%update Y
fprintf('\nFirst Iteration for Updating Y\n');
Y=zeros(n,c);
tF = zeros(n, c);
for i = 1:V
[v,d] = eig(L{i});
d = diag(d);
[~, idx] = sort(d);
F{i} = v(:,idx(1:c));
end
fprintf('\nFirst Iteration for Updating F\n');
for i = 1:V
    tF = tF + w(1,i) * D{i}^0.5 * F{i};
end

[~,I]=max(tF,[],2);
for i=1:n
    Y(i,I(i,1))=1;
end
%update F{i}
tic;
parfor i = 1:V
[F{i},~]= solveF(F{i},@fun1,opts,lambda1,Y,D{i},L{i});
end
toc

%update w
fprintf('\nFirst Iteration for Updating w\n');
e = zeros(1, V);
for i=1:V
    e(1,i)=trace(F{i}' * L{i} * F{i}) - lambda1 * trace((D{i}^0.5 * F{i})' * Y);
end
ad = e/(2*lambda2);
w= EProjSimplex_new(-ad);

p=0;
for i=1:V
   p=p+w(1,i)*(trace(F{i}' * L{i} * F{i}) - lambda1 * trace((D{i}^0.5 * F{i})' * Y)); 
end
p=p+lambda2*trace(w * w');
p_iter(1,1)=p;
%% ------------------- 3. all iteration ------------------
for t = 2:100
    fprintf('\nUpdating Y\n');
    Y=zeros(n,c);
    tF = zeros(n, c);  
    for i = 1:V
        tF = tF + w(1,i) * D{i}^0.5 * F{i};
    end
    [~,I]=max(tF,[],2);
    for i=1:n
        Y(i,I(i,1))=1;
    end
    
    [~, predY] = max(Y, [], 2);
    result = ClusteringMeasure(gnd, predY)

    fprintf('\nUpdating F\n');
    tic;
    parfor i = 1:V
        [F{i},~]= solveF(F{i},@fun1,opts,lambda1,Y,D{i},L{i});
    end
    toc
    fprintf('\nUpdating w\n');
    e = zeros(1, V);
    for i=1:V
        e(1,i)=trace(F{i}' * L{i} * F{i}) - lambda1 * trace((D{i}^0.5 * F{i})' * Y);
    end
    ad = e/(2*lambda2);
    w= EProjSimplex_new(-ad);
    
    p=0;
    for i=1:V
       p=p+w(1,i)*(trace(F{i}' * L{i} * F{i}) - lambda1 * trace((D{i}^0.5 * F{i})' * Y)); 
    end
    p=p+lambda2*trace(w * w');
    p_iter(1,t)=p;

    if t>2 && (abs(p_iter(t)-p_iter(t-1))/p_iter(t-1)) < 1e-6
        fprintf('\nAlgorithm Convergence\n');
        break;
    end
end

end

function [F,G]=fun1(P,alpha,Y,D,L)
    G=2*L*P-alpha*D^0.5*Y;
    F=trace(P'*L*P)-alpha * trace(P'*D^0.5*Y);
end