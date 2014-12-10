
clear all; close all; clc;

% load full training data
load('../data/train.mat');
totnum=size(digitdata,1);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);
rng(0); %so we know the permutation of the training data
randomorder=randperm(totnum);

% randomize the full training set
digitdata = digitdata(randomorder,:);
targets = targets(randomorder,:);

% split the train and validation data
Ntrain = 50000;
traindata = digitdata(1:Ntrain,:);
traintargets = targets(1:Ntrain,:);
train_batchsize = 20;
[train_batchdata,train_batchtargets] = makebatchdata(traindata,traintargets,train_batchsize) ;
train_batchdata = single(train_batchdata);
train_batchtargets = single(train_batchtargets);

% split the full training data into train data and validation data
Nval = 10000;
valdata = digitdata(Ntrain+1:Ntrain+Nval,:);
valtargets = targets(Ntrain+1:Ntrain+Nval,:);
val_batchsize = Nval;
[val_batchdata,val_batchtargets] = makebatchdata(valdata,valtargets,val_batchsize) ;
val_batchdata = single(val_batchdata);
val_batchtargets = single(val_batchtargets);

clear digitdata targets valdata valtargets traindata traintargets;

load('../data/test.mat');
Ntest = 10000;
test_batchsize = Ntest;
[test_batchdata,test_batchtargets] = makebatchdata(digitdata,targets,test_batchsize) ;
test_batchdata = single(test_batchdata);
test_batchtargets = single(test_batchtargets);

disp('size of train i/o data');
[~,din,train_numbats] = size(train_batchdata)
[~,dout,train_numbats] = size(train_batchtargets)

disp('size of validation i/o data');
[~,~,val_numbats] = size(val_batchdata)
[~,~,val_numbats] = size(val_batchtargets)

disp('size of test i/o data');
[~,~,test_numbats] = size(test_batchdata)
[~,~,test_numbats] = size(test_batchtargets)

% nn params settings
numepochs = 100*(train_batchsize);
mf = 0.9; % momentum factor
berp = [1 0.5 1]; % bernoulli prob of output layer is always 1
mncflag = 0;
mnc = 4;

arch_name = strcat('500N10M','_bs',num2str(train_batchsize));
nl = [din 500 dout];
nh = length(nl) - 1; % number of hidden layers
f = [ 'R' 'M'];

if (length(nl) - 1) ~= length(f)
    disp('number of hidden o/p fns mus be same as number of hidden layers');
end

% weight initialization
nlv = 1:nh;
wtl = [1 nl(nlv).*nl(nlv+1)];
wtl = cumsum(wtl);
btl = cumsum([1 nl(nlv+1)]);

W = zeros(1,sum(nl(1:end-1).*nl(2:end)));
b = zeros(1,sum(nl(2:end)));

winit_meth = 'yoshua';

switch winit_meth
    case 'kp_init'       
        for i = 1:nh
            maxweight = 3/sqrt(nl(i));
            W(1,wtl(i):wtl(i+1)-1) = 2*maxweight*rand(1,nl(i)*nl(i+1)) - maxweight;
        end        
    otherwise
        for i = 1:nh-1
            maxweight = sqrt(6/(nl(i) + nl(i+1)));
            W(1,wtl(i):wtl(i+1)-1) = 2*maxweight*rand(1,nl(i)*nl(i+1)) - maxweight;
        end
end

Gb = gpuArray(b);
GW = gpuArray(W);
pdW = zeros(size(W));
pdb = zeros(size(b));
GpdW = gpuArray(pdW);
Gpdb = gpuArray(pdb);

disp('size of weight matrix');
size(W)

%%%%%%%%%%%%%%%%%%%%%%%%% done with initialization %%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%% actual ANN training starts here %%%%%%%%%%%%%%%%

wtdir = '../wt/';
mkdir(wtdir);
traindnn

% ann_train(GW,Gb,GpdW,Gpdb,nl,f,numepochs,mf,train_batchdata,train_batchtargets, ...
%     Ntrain,val_batchdata,val_batchtargets,Nval,arch_name,berp,mncflag,mnc)