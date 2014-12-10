function [ W,b,dW,db] = bp_mf_av_mnist(X,W,b,nl,fl,ol,Out,pdW,pdb,mf,bs,aeta,otl,wtl,btl,nh,a_tanh,b_tanh,bby2a,cfn,l1,l2)

% bp - Back Propogation learning law
% Input : (1) X  - Input data matrix (1 x din)
%         (2) Wt - Weight cell (Weights at all layers including biases)
%         (3) nl - Num of nodes at each layer (1 x (nh+1) )
%         (4) fl - Function at each layer ('N' or 'S' or 'L')
%         (5) ol - Output cell (outputs at each layer)
%         (6) ac - Activation cell (activation values at each layer)
%         (7) Out - Activation cell (actual output)

% Output : (1) W - Updated weight cell


% Step 1 : Compute derivative of error fn (E) w.r.t wts

% updating weights of top most layer
ol_m = reshape(ol(1,otl(end-1):otl(end)-1),bs,nl(end));
ol_pl_m = reshape(ol(1,otl(end-2):otl(end-1)-1),bs,nl(end-1));

switch fl(end)
    case 'N'
        der_f = bby2a*((a_tanh - ol_m).*(a_tanh + ol_m));
    case 'S'
        der_f = b_tanh*(ol_m.*(1 - ol_m));
    case 'R' % added on 28/11/14
        der_f = ones(bs,nl(end)).*(ol_m > 0);
    case 'M' % Softmax layer
        der_f = (ol_m.*(1 - ol_m));
    case 'L'
        der_f = ones(bs,nl(end));
    otherwise
        disp('please enter a valid output function name (N/S/R/M/L)');
        return;
end

switch cfn
    case 'ls'
        costder = -(Out - ol_m);
        del_bp = costder.*der_f;
    case  'nll'
%         costder = -(Out./ol_m);
        del_bp  = -(Out - ol_m);
end

own = reshape(W(1,wtl(nh):wtl(nh+1)-1),nl(nh+1),nl(nh))';
% del_bp = costder.*der_f;
dbv = -aeta(end)*sum(del_bp,1)/bs;
dWm = -aeta(end)*((ol_pl_m'*del_bp)/bs + l1*sign(own) + l2*2*own);

db = gpuArray(zeros(1,btl(end-1)));
db(1,btl(nh):btl(end)-1) = dbv;
dW = gpuArray(zeros(1,wtl(end)-1));
dW(1,wtl(nh):wtl(nh+1)-1) = reshape(dWm',1,numel(dWm));

% updating weights of inner hidden layers
for j = nh-1:-1:1
    
    ol_m = reshape(ol(1,otl(j):otl(j+1)-1),bs,nl(j+1));
    
    if (j-1) ~=0
        ol_pl_m = reshape(ol(1,otl(j-1):otl(j)-1),bs,nl(j));
    else
        ol_pl_m = X;
    end
    
    switch fl(j)
        case 'N'
            der_f = bby2a*((a_tanh - ol_m).*(a_tanh + ol_m));
        case 'S'
            der_f = b_tanh*(ol_m.*(1 - ol_m));
        case 'R' % added on 28/11/14
            der_f = ones(bs,nl(j+1)).*(ol_m > 0);
        case 'M' % Softmax layer
            der_f = (ol_m.*(1 - ol_m));
        case 'L'
            der_f = ones(bs,nl(j+1));
        otherwise
            disp('error: please enter a valid output function name (N/S/R/M/L)');
            return;
    end
    
    wdel_bp = del_bp*(own'); % weghted del_bps
    del_bp = wdel_bp.*der_f;
    dbv = -aeta(j)*sum(del_bp,1)/bs;
    own = reshape(W(1,wtl(j):wtl(j+1)-1),nl(j+1),nl(j))'; % only weights are considered , ingore the first row which are biases
    dWm = -aeta(j)*((ol_pl_m'*del_bp)/bs + l1*sign(own) + l2*2*own);
    
    db(1,btl(j):btl(j+1)-1) = dbv;
    dW(1,wtl(j):wtl(j+1)-1) = reshape(dWm',1,numel(dWm));
end

db = db + mf*pdb;
dW = dW + mf*pdW;

% Step 2 : Update weights
b = b + db;
W = W + dW;


end

