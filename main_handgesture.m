clear all; clc;
addpath('../ext_func');
%% import testing and training data on product Grassmann manifold 
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
% parameter setting
omega = [1/3,1/3,1/3]; % importance of each factor manifold
Solver_Flag = 1;  %1: SPAMS, 2: CVX

%% computing CRR with different parameter p
p1 = 1:20;
p2 = 1:20;
p3 = 1:20;
for i = 1 : length(p1)
    for j = 1 : length(p2)
         for k = 1 : length(p3)
             for num = 1 : 4
                 Data_test_sub.data{1} = Data_test.data{num,1}(:,1:p1(i),:); 
                 Data_test_sub.data{2} = Data_test.data{num,2}(:,1:p2(j),:); 
                 Data_test_sub.data{3} = Data_test.data{num,3}(:,1:p3(k),:); 
                 Data_test_sub.label = Data_test.label{num}; 
                 Data_train_sub.data{1} = Data_train.data{1}(:,1:p1(i),:); 
                 Data_train_sub.data{2} = Data_train.data{2}(:,1:p2(j),:); 
                 Data_train_sub.data{3} = Data_train.data{3}(:,1:p3(k),:); 
                 for SR_lambda = [0.1]   %sparse representation parameter
                   [gSC_alpha,gSC_qX,gSC_D] = pgsc_func(Data_test_sub,Data_train_sub,SR_lambda,Solver_Flag,omega);
                   y_hat = Classify_SRC(gSC_D,Data_train.label,gSC_alpha,gSC_qX);
                   CRR(i,j,k,num) = sum(double(y_hat == Data_test_sub.label'))/length(Data_test_sub.label);
                 end
                 clear Data_test_sub
                 clear Data_test_sub
                 clear  Data_train_sub
             end       
         end
     end
end
save('CRR_omega(0.33_0.33_0.33).mat','CRR');
%% plot p1 p2 p3
load('CRR_omega(0.33_0.33_0.33).mat');% imporrt the saved data
p1 = 1:20;
p2 = 1:20;
p3 = 1:20;
for i = 1 : 4 
    CRR_subset{i} = CRR(:,:,:,i);
    max_CRR = max(CRR_subset{i}(:));
    s = size(CRR_subset{i});
    Lax = find(CRR_subset{i} >= max_CRR);
    [m,n,t]=ind2sub(s,Lax);
    p_star = [p1(m(1));p2(n(1));p3(t(1))]';
    figure(i)
    x = 1 : 20; 
    curve1 = CRR_subset{i}(:,p_star(2),p_star(3));
    curve2 = CRR_subset{i}(p_star(1),:,p_star(3));
    curve3 = reshape(CRR_subset{i}(p_star(1),p_star(2),:),[20,1]);
    plot(x,curve1,'-*',x,curve2,':.',x,curve3,'-.+');
    hold on
    xlabel('dimension of each factor Grassmann manifold');
    ylabel('CRR');
    title(['Set',num2str(i)]);
    legend('p_1','p_2','p_3');
end
%% finding p_star
load('CRR_omega(0.33_0.33_0.33).mat');% imporrt the saved data
p1 = 1:20;
p2 = 1:20;
p3 = 1:20;
CRR_mean = CRR(:,:,:,1)+ CRR(:,:,:,2)+ CRR(:,:,:,3)+ CRR(:,:,:,4);
max_CRR = max(CRR_mean (:));
s = size(CRR_mean);
Lax = find(CRR_mean >= max_CRR);
[m,n,t]=ind2sub(s,Lax);
p_star = [p1(m(1));p2(n(1));p3(t(1))]';
%%  computing CRR with different  omega
p1 = 1:20;
p2 = 1:20;
p3 = 1:20;
i = 7; j = 2; k = 20;  
for num = 1 : 4
    Data_test_sub.data{1} = Data_test.data{num,1}(:,1:p1(i),:); 
    Data_test_sub.data{2} = Data_test.data{num,2}(:,1:p2(j),:); 
    Data_test_sub.data{3} = Data_test.data{num,3}(:,1:p3(k),:); 
    Data_test_sub.label = Data_test.label{num}; 
    Data_train_sub.data{1} = Data_train.data{1}(:,1:p1(i),:); 
    Data_train_sub.data{2} = Data_train.data{2}(:,1:p2(j),:); 
    Data_train_sub.data{3} = Data_train.data{3}(:,1:p3(k),:); 
    h1 = 0;
    CRR{num} = zeros(101,101);
    for omega1 = 0:0.01:1
        h1 = h1 + 1;
        h2 = 0;
         for omega2 = 0:0.01:(1-omega1)
             h2 = h2 + 1;
             omega = [omega1 omega2  1-omega1-omega2 ];
             for SR_lambda = [0.1]  %sparse representation parameter
                   [gSC_alpha,gSC_qX,gSC_D] = pgsc_func(Data_test_sub,Data_train_sub,SR_lambda,Solver_Flag,omega);
                   y_hat = Classify_SRC(gSC_D,Data_train.label,gSC_alpha,gSC_qX);
                   CRR{num}(h1,h2) = sum(double(y_hat == Data_test_sub.label'))/length(Data_test_sub.label);
             end     
         end
    end
    clear Data_test_sub
end
save('CRR_omega_all.mat','CRR')
%% plot omega
load CRR_omega_all.mat; % imporrt the saved data
for k = 1 : 4
    [omega1,omega2] = meshgrid(0:0.01:1,0:0.01:1.0);
    [row,col]  = find(CRR{k}==0);
    CRR1 = CRR{k};
    for i = 1 : length(row)
       omega1(row(i),col(i)) = 0;
       omega2(row(i),col(i)) = 0;
       CRR1(row(i),col(i)) = 0.6;
    end
    for i = 1 : 101
        for j = 1 : 101
            if CRR1(i,j) < 0.6
                CRR1(i,j) = 0.6;
            end
        end
    end
    figure(k);
    h = pcolor(CRR1);
    axis normal
    xlabel(' \omega_1(10^{-2})');
    ylabel(' \omega_2(10^{-2})');
    title(['Set',num2str(k)]);
    set(h,'edgecolor','none','facecolor','interp');
    colorbar;
    clear CRR1;
end
%% computing CRR_mean_omega 
CRR_mean_omega = CRR{1} + CRR{2} + CRR{3} + CRR{4};
max_CRR = max(CRR_mean_omega (:));
s = size(CRR_mean_omega);
Lax = find(CRR_mean_omega >= max_CRR);
[m,n] = ind2sub(s,Lax);
omega_star = [omega1(1,m(1));omega2(n(1),1)]';

 



