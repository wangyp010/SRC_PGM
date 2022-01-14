clear all; clc;
addpath('../ext_func');
%% import data
load Set1PGM_gray.mat
Data_test.data{1,1} = PGM.data1; 
Data_test.data{1,2} = PGM.data2;
Data_test.data{1,3} = PGM.data3;
Data_test.label{1} = PGM.label;
clear PGM
load Set2PGM_gray.mat
Data_test.data{2,1} = PGM.data1; 
Data_test.data{2,2} = PGM.data2;
Data_test.data{2,3} = PGM.data3;
Data_test.label{2} = PGM.label;
clear PGM
load Set3PGM_gray.mat
Data_test.data{3,1} = PGM.data1; 
Data_test.data{3,2} = PGM.data2;
Data_test.data{3,3} = PGM.data3;
Data_test.label{3} = PGM.label;
clear PGM
load Set4PGM_gray.mat
Data_test.data{4,1} = PGM.data1; 
Data_test.data{4,2} = PGM.data2;
Data_test.data{4,3} = PGM.data3;
Data_test.label{4} = PGM.label;
clear PGM
load Set5PGM_gray.mat
Data_train.data{1} = PGM.data1; 
Data_train.data{2} = PGM.data2;
Data_train.data{3} = PGM.data3;
Data_train.label = PGM.label;
clear PGM
%% parameter
Solver_Flag = 1;  %1: SPAMS, 2: CVX

%%  5-fold cross validation on Set5, to select p1,p2,p3,\omega_!,\omega_2,\omega_3
p1 = 2:2:20; p2 =  2:2:20; p3 =  2:2:20;
p = cell(1,length(p1)*length(p2)*length(p3));
n_p = 0;
for i = 1 : length(p1)
    for j = 1 : length(p2)
         for k = 1 : length(p3)
             n_p = n_p +1;
             p{n_p} = [p1(i),p2(j),p3(k)]; 
         end
    end
end

data_r= size(Data_train.data{1,1},3);
K = 5;
indices = crossvalind('Kfold', data_r, K);
% CRR_joint_hand = [];
% Data_test_sub = cell(1,size(p,2));
% Data_train_sub = cell(1,size(p,2));
% tic
% parfor i = 1 : size(p,2)
%     i
%      for d = 1 : K
%                  test = (indices == d);
%                     % 取反，获取第i份训练数据的索引逻辑值
%                   train = ~test;   
%                   Data_test_sub{i}.data{1} = Data_train.data{1,1}(:,1:p{i}(1),test); 
%                   Data_test_sub{i}.data{2} = Data_train.data{1,2}(:,1:p{i}(2),test); 
%                   Data_test_sub{i}.data{3} = Data_train.data{1,3}(:,1:p{i}(3),test); 
%                   Data_test_sub{i}.label = Data_train.label(test);
% 
%                   Data_train_sub{i}.data{1} = Data_train.data{1,1}(:,1:p{i}(1),train); 
%                   Data_train_sub{i}.data{2} = Data_train.data{1,2}(:,1:p{i}(2),train); 
%                   Data_train_sub{i}.data{3} = Data_train.data{1,3}(:,1:p{i}(3),train); 
%                   Data_train_sub{i}.label =  Data_train.label(train);
%                  for omega1 = 0.1:0.1:0.8
%                      for omega2 = 0.1:0.1: 0.95-omega1
%                           omega = [omega1 omega2  1-omega1-omega2 ];
%                          for SR_lambda = [0.1] 
%                                [gSC_alpha,gSC_qX,gSC_D] = pgsc_func(Data_test_sub{i},Data_train_sub{i},SR_lambda,Solver_Flag,omega);
%                                y_hat = Classify_SRC(gSC_D,Data_train_sub{i}.label,gSC_alpha,gSC_qX);
%                                CRR_joint_hand = [CRR_joint_hand sum(double(y_hat == Data_test_sub{i}.label'))/length(Data_test_sub{i}.label)];
%                          end    
%                      end
%                  end
%     end
% end
% toc
% save('CRR_joint_hand_5fold.mat','CRR_joint_hand');
%%
load('CRR_joint_hand_5fold.mat');
K = 5;
rr = 0;
for i = 1 : length(p1)
         for j = 1 : length(p2)
             for k = 1 : length(p3) 
                 for d = 1 : K
                     h1 = 0;
                     for omega1 = 0.1:0.1:0.8
                         h1 = h1 + 1;
                         h2 = 0;
                         for omega2 = 0.1: 0.1: 0.95-omega1
                             rr = rr + 1;
                             h2 = h2 + 1;
                             CRR_joint_hand_reshape(i,j,k,h1,h2,d) = CRR_joint_hand(rr);
                         end
                     end
                 end
             end
         end
end
% 重排成一个矩阵，行为所有参数，列为交叉验证的10个
omega1 = 0.1:0.1:0.8;
omega2 = 0.1:0.1:0.8;
nn = 0;
for i = 1 : length(p1)
     for j = 1 : length(p2)
         for k = 1 : length(p3) 
             for h1 = 1 : 8
                 for h2 = 1 : 8
                     nn = nn + 1;
                     for d = 1 : K
                         CRR_matrix(nn,d )  = CRR_joint_hand_reshape(i,j,k,h1,h2,d);
                     end
                     parameter{nn} = [p1(i),p2(j),p3(k),omega1(h1),omega2(h2)];
                 end
             end
         end
     end
end

% computing p_star,omega_star
CRR_opt = sum(CRR_joint_hand_reshape,6);
max_CRR_joint = max(CRR_opt(:));
s = size(CRR_opt);
Lax = find(CRR_opt >= max_CRR_joint);
[m,n,t,h,f] = ind2sub(s,Lax);

for tt = 1 : size(m,1)
    CRR_opt_shaixuan(tt,:) = CRR_joint_hand_reshape(m(tt),n(tt),t(tt),h(tt),f(tt),:);
    p_star_ori(tt,:) = [p1(m(tt)),p2(n(tt)),p3(t(tt))]';
    omega_star_ori(tt,:) =[omega1(h(tt)),omega2(f(tt)),1-omega1(h(tt))-omega2(f(tt))];
end
p_sum = sum(p_star_ori,2);
id= find(p_sum >= prctile(p_sum,95))';
nn =1;
for kk = id(1)
    p_star(kk,:) = [p1(m(kk)),p2(n(kk)),p3(t(kk))]';
    omega_star(kk,:) =[omega1(h(kk)),omega2(f(kk)),1-omega1(h(kk))-omega2(f(kk))];
    %testing on Set1-Set4
    clear Data_test_sub
    clear  Data_train_sub
    for num = 1 : 4
         Data_test_sub.data{1} = Data_test.data{num,1}(:,1:p_star(kk,1),:); 
         Data_test_sub.data{2} = Data_test.data{num,2}(:,1:p_star(kk,2),:); 
         Data_test_sub.data{3} = Data_test.data{num,3}(:,1:p_star(kk,3),:); 
         Data_test_sub.label = Data_test.label{num}; 
         Data_train_sub.data{1} = Data_train.data{1}(:,1:p_star(kk,1),:); 
         Data_train_sub.data{2} = Data_train.data{2}(:,1:p_star(kk,2),:); 
         Data_train_sub.data{3} = Data_train.data{3}(:,1:p_star(kk,3),:); 
         for SR_lambda = [0.1]  
           tic
           [gSC_alpha,gSC_qX,gSC_D] = pgsc_func(Data_test_sub,Data_train_sub,SR_lambda,Solver_Flag,omega_star(kk,:));
           y_hat = Classify_SRC(gSC_D,Data_train.label,gSC_alpha,gSC_qX);
           CRR_test(kk,num) = sum(double(y_hat == Data_test_sub.label'))/length(Data_test_sub.label);
           toc
         end
         % confusion matrix
         classLabels = ['FL';'FR';'FC';'SL';'SR';'SC';'VL';'VR';'VC'];
         [Matr,order] = confusionmat(Data_test_sub.label',y_hat);
         figure(nn)
         cm = confusionchart(Matr,classLabels,'FontSize',20);
         cm.XLabel = 'Predicted Class';
         cm.YLabel = 'True Class';
         %cm.Title = ['Confusion matrix of set',num2str(num)];
         nn = nn + 1;
    end  
end
%% CRR of five folds on training set (Set5)
for kk = id
    for d = 1 : K
       CRR_cross_validation(kk,d) = CRR_joint_hand_reshape(m(kk),n(kk),t(kk),h(kk),f(kk),d);
    end
end






