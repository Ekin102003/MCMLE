clear all
currentFolder = pwd;
addpath(genpath(currentFolder));


load BBC.mat;

X = data;
Y = truelabel{1};

for i = 1:length(X)
    X{i} = X{i}';
    X{i} = tfidf(X{i});
end
cluser_num = length(unique(Y));

for k = 5
for i = 1:length(X)
% ---------- initilization for Z and F -------- %
options = [];
options.NeighborMode = 'KNN';
options.k = k;
options.WeightMode = 'HeatKernel';      % Binary  HeatKernel
W = constructW(X{i},options);
W = full(W);
Z1 = W-diag(diag(W));         
W = (Z1+Z1')/2;
D{i}= diag(sum(W));     
L0 = D{i} - W;
L{i} = D{i}^(-0.5)*L0*D{i}^(-0.5);
end


lambda1list = 0.0005;%[0.0001 0.001 0.01 0.1 1 10 100 1000];
lambda2list = 10;%[0.1 1 10 100 1000 10000];

for rtimes = 1:1
    for i = 1:length(lambda1list)
        for j = 1:length(lambda2list)
        lambda1 = lambda1list(i);
        lambda2 = lambda2list(j);
        [~,G,s,t] = MCMLE(L,D,cluser_num,lambda2,length(L),lambda1,length(Y),Y);
%G = MVSpectralClustering(Ln, cluser_num, r, 'nmf');
[~, predY] = max(G, [], 2);
result = ClusteringMeasure(Y, predY)
%dlmwrite('BBC_2021_1217.txt', [k, lambda1, lambda2, t, result],'-append','delimiter','\t','newline','pc');
        end
    end


   
end
end