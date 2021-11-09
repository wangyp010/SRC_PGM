function [alpha,Y_star,A] = pgsc_func(Data_test,Data_train,SR_lambda,Solver_Flag,omega)

[~,p(1),nAtoms] = size(Data_train.data{1}); % nAtoms is the number of train samples
[~,p(2),~] = size(Data_train.data{2}); 
[~,p(3),~] = size(Data_train.data{3}); 
nPoints = size(Data_test.data{1},3);% number of test samples
% line 1-8 in Algorithm 1
for m = 1 : length(p)
    KX{m} = grassmann_proj(Data_train.data{m});
    KXY{m} = grassmann_proj(Data_test.data{m},Data_train.data{m});
end
% line 9 in Algorithm 1
KX_sum = omega(1)*KX{1}+omega(2)*KX{2}+omega(3)*KX{3};
[KX_U,KX_Sigma,~] = svd(KX_sum);
% line 10 in Algorithm 1
A = diag(sqrt(diag(KX_Sigma)))*KX_U';
% line 11 in Algorithm 1
Y_U = KX_U*diag(1./sqrt(diag(KX_Sigma)));
KXY_sum = omega(1)*KXY{1} + omega(2)*KXY{2} +omega(3)*KXY{3};
Y_star = Y_U'*KXY_sum;
% line 12 in Algorithm 1
switch Solver_Flag
    case 1
        alpha = full(mexLasso(Y_star,A,struct('mode',2,'lambda',SR_lambda,'lambda2',0)));
    otherwise 
        alpha = zeros(nAtoms,nPoints);       
        for tmpC1 = 1:nPoints 
            cvx_begin quiet;
            variable alpha_cvx(nAtoms,1);
            minimize( norm(Y_star(:,tmpC1) - A * alpha_cvx) +  SR_lambda*norm(alpha_cvx,1));
            cvx_end;
            alpha(:,tmpC1) = double(alpha_cvx);
        end  
end
end

function dist_p = grassmann_proj(SY1,SY2)

MIN_THRESH = 1e-6;

same_flag = false;
if (nargin < 2)
    SY2 = SY1;
    same_flag = true;
end
p = size(SY1,2);


[~,~,number_sets1] = size(SY1);
[~,~,number_sets2] = size(SY2);

dist_p = zeros(number_sets2,number_sets1);

if (same_flag)
    %SY1 = SY2
    for tmpC1 = 1:number_sets1
        Y1 = SY1(:,:,tmpC1);
        for tmpC2 = tmpC1:number_sets2
            tmpMatrix = Y1'* SY2(:,:,tmpC2);
            tmpProjection_Kernel_Val = sum(sum(tmpMatrix.^2));
            if (tmpProjection_Kernel_Val < MIN_THRESH)
                tmpProjection_Kernel_Val = 0;
            elseif (tmpProjection_Kernel_Val > p)
                tmpProjection_Kernel_Val = p;
            end
            dist_p(tmpC2,tmpC1) = tmpProjection_Kernel_Val;
            dist_p(tmpC1,tmpC2) = dist_p(tmpC2,tmpC1);
        end
    end
else
    for tmpC1 = 1:number_sets1
        Y1 = SY1(:,:,tmpC1);
        for tmpC2 = 1:number_sets2
            tmpMatrix = Y1'* SY2(:,:,tmpC2);
            tmpProjection_Kernel_Val = sum(sum(tmpMatrix.^2));
            if (tmpProjection_Kernel_Val < MIN_THRESH)
                tmpProjection_Kernel_Val = 0;
            elseif (tmpProjection_Kernel_Val > p)
                tmpProjection_Kernel_Val = p;
            end
            dist_p(tmpC2,tmpC1) = tmpProjection_Kernel_Val;
        end
    end
end


end