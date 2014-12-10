
addpath('../../DNN_SPS/nn/GPUver/arraystyle');

% open error text file
fid = fopen(strcat('../wt/err_',arch_name,'.err'),'w');

% initialze error vector
valerr = zeros(numepochs,1);
valerr = gpuArray(valerr);

% load data onto GPU
train_batchdata = gpuArray(train_batchdata);
train_batchtargets = gpuArray(train_batchtargets);
val_batchdata = gpuArray(val_batchdata);
val_batchtargets = gpuArray(val_batchtargets);
test_batchdata = gpuArray(test_batchdata);
test_batchtargets = gpuArray(test_batchtargets);

% train set variables
otl = [1 train_batchsize*(nl(nlv+1))];
otl = cumsum(otl);

% % test error set variables
ottl = [1 Ntest*(nl(nlv+1))];
ottl = cumsum(ottl);

% validation set variables
otvl = [1 Nval*(nl(nlv+1))];
otvl = cumsum(otvl);

% initialize nonlinearity and learning rate per layer params
l1 = 0;
l2 = 0;
cfn = 'nll';
a_tanh = 1.7159;
b_tanh = 2/3;
bby2a = (b_tanh/(2*a_tanh));
aeta = [1./(25*nl(1:end-2)) 1./(25*sum(nl(2:end)))];
aeta = [0.01 0.01];

% early stopping params (Theano DeepLearningTutorials)
patience = 10000;
patience_inc = 2;
imp_thresh = 0.995;
val_freq = min(train_numbats,patience/2);
best_val_loss = inf;
best_iter = 0;


for NE = 1:numepochs
    
    % for each epoch
    tne = tic;
    iter = (NE-1)*train_numbats;
    
    tvde = tic;
    % Validation data error computation
    for i = 1:val_numbats
        % fp
        [ol] = fp_av_test_mnist(val_batchdata(:,:,i),GW,Gb,nl,f,nh,a_tanh,b_tanh,wtl,btl,berp,val_batchsize);
        
        % compute error
        ol_mat = reshape(ol(1,otvl(end-1):otvl(end)-1),val_batchsize,nl(end));
        clear ol;
        [me] = compute_zerooneloss(ol_mat,val_batchtargets(:,:,i));
        clear ol_mat;
        valerr(NE) = valerr(NE) + me/val_numbats;
    end
    toc(tvde)
    
    % Print error (validation) per epoc
    fprintf('Epoch : %d  Val Loss : %f \n',NE,valerr(NE)*100);
    
    if valerr(NE) < best_val_loss
        if valerr(NE) < (best_val_loss*imp_thresh)
            patience = max(patience,iter*patience_inc);
        end
        best_val_loss = valerr(NE);
        best_iter = iter;
        
        %ttde = tic;
        testerr = 0;
        % Test data error computation
        for i = 1:test_numbats
            % fp
            [ol] = fp_av_test_mnist(test_batchdata(:,:,i),GW,Gb,nl,f,nh,a_tanh,b_tanh,wtl,btl,berp,test_batchsize);
            
            % compute error
            ol_mat = reshape(ol(1,ottl(end-1):ottl(end)-1),test_batchsize,nl(end));
            clear ol;
            me = compute_zerooneloss(ol_mat,test_batchtargets(:,:,i));
            clear ol_mat;
            testerr = testerr + me/test_numbats;
        end
        %toc(ttde)
        
        % Print error (validation and testing) per epoc
        fprintf(fid,'%d %f %f \n',NE,valerr(NE)*100,testerr*100);
        
        % Print error (testing) per epoc
        fprintf('\t Epoch : %d  Test Loss : %f \n',NE,testerr*100);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%% save weight file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % save parameters every epoch
        W = gather(GW); b = gather(Gb);
        save(strcat('../wt/W_',arch_name,'.mat'),'W','b');
        pdW = gather(GpdW); pdb = gather(Gpdb);
        save(strcat('../wt/pdW_',arch_name,'.mat'),'pdW','pdb');
    end
    
    % Weight update step
    twu = tic;
    
    % fp and bp for each batch
    for i = 1:train_numbats
        
        % fp
        [ol] = fp_av_mnist(train_batchdata(:,:,i),GW,Gb,nl,f,nh,a_tanh,b_tanh,wtl,btl,berp,train_batchsize);
        
        % bp
        [ GW,Gb,GpdW,Gpdb] = bp_mf_av_mnist(train_batchdata(:,:,i),GW,Gb,nl,f, ...
            ol,train_batchtargets(:,:,i),GpdW,Gpdb,mf,train_batchsize,aeta,otl,wtl,btl,nh,a_tanh,b_tanh,bby2a,cfn,l1,l2);
        
        
%         if mncflag
%             for j = 1:nh
%                 rWt = reshape(GW(1,wtl(j):wtl(j+1)-1),nl(j+1),(nl(j)+1))';
%                 lenwv{j} = sum(rWt(2:end,:).^2)'; % weight vector lengths excluding biases
%                 idxin = find(lenwv{j} > mnc);
%                 
%                 if ~isempty(idxin)
%                     rWt(2:end,idxin) = bsxfun(@rdivide,rWt(2:end,idxin),(lenwv{j}(idxin))'/mnc);
%                 end
%                 mlenwv{j} = sum(rWt(2:end,:).^2)'; % modified weight vector lengths excluding biases
%                 
%                 % replace the original weights with the modified weights
%                 GW(1,wtl(j):wtl(j+1)-1) = reshape(rWt',1,(nl(j)+1)*nl(j+1));
%             end
%         end
        
    end
    clear ol
    toc(twu)
    
    % print elapsed time
    fprintf('Elapse Time: %f \n',toc(tne));
    
end

fclose(fid);

fprintf('Training done !!!')
fprintf('Best val error : %f ; at epoch : %d ; with test error : %f', best_val_loss,floor(best_iter/train_numbats),testerr)
