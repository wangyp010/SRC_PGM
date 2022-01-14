clear all; clc;
load('CRR_joint_hand_5fold.mat')
%% plot p1 p2 p3
p1 = 2:2:20;
p2 = 2:2:20;
p3 = 2:2:20;

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
% rearrange to a matrix,rows:all parameters; col: number of cross validation sets
omega1 = 0.1:0.1:0.8;
omega2 = 0.1:0.1:0.8;
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

%----------plot p1,p2,p3------------------------
nn =1;
for kk = id
    figure(nn)
    curve1 = CRR_opt(:,n(kk),t(kk),h(kk),f(kk))/K;
    curve2 = CRR_opt(m(kk),:,t(kk),h(kk),f(kk))/K;
    curve3 = reshape(CRR_opt(m(kk),n(kk),:,h(kk),f(kk)),[length(p3),1])/K;
    plot(p1,curve1,'-r',p2,curve2,':g',p3,curve3,'-.b','linewidth',2);
    set(gca,'FontSize',12);
    hold on
    xlabel('dimension of each factor Grassmann manifold','FontSize',15);
    ylabel('the mean CRR of 5 crossvalid sets','FontSize',15);
    title(['Combination',num2str(nn)],'FontSize',15);
    h1 = legend('p_1','p_2','p_3');
    set(h1,'location','southeast','Fontname', 'Times New Roman','FontWeight','bold','FontSize',15)    
    nn = nn + 1;
end

%% ------------ plot omega---------------
mm = 4;
for kk = id
    figure(mm)
    x = 0.1:0.1:0.8;
    CRR_omega = reshape(CRR_opt(m(kk),n(kk),t(kk),:,:)/K,[8,8]);
    b = bar3(CRR_omega);
    set(gca,'xticklabel',x,'yticklabel',x,'FontSize',10)
    zlim([0.95,1]);
    xlabel('\omega_2','FontSize',15);ylabel('\omega_1','FontSize',15);zlabel('CRR','FontSize',15)
    title(['Combination',num2str(mm-3)],'FontSize',15);
    mm = mm + 1;
end




