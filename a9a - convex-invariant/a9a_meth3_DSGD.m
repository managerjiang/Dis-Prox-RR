clear all
%次梯度固定步长
%% 数据加载
load('a9a_smote.mat');%载入A：A(48243x123):48243个数据
load('L_a9a_smote.mat');%载入L：A(1x48243):48243个结果
A=A1;
L=L1;
A=double(A);
L=double(L);
L(L==0)=-1;
L(L==1)=1;%由于正负样本比例是1:4
%% 参数设置
agent_num=10;% agent个数
Maxgen=200;% 迭代次数

load('data/C_meth1_smote_800.mat');%载入C_store
C=C_store;
% x_k_i_last=zeros(123,agent_num);% 
q=zeros(123,agent_num);% Q阵
lamuda1=0.5*10^(-5);
lamuda2=0.5*10^(-5);
global v;% V阵
%% 数据预处理
%根据智能体个数裁剪数据，每个智能体十分之一的数据
for i=1:agent_num
    L_cut(i,:)=L((i-1)*floor(size(A,1)/agent_num)+1:i*floor(size(A,1)/agent_num));
    A_cut(:,:,i)=A((i-1)*floor(size(A,1)/agent_num)+1:i*floor(size(A,1)/agent_num),:); 
end
%% GUROBI设置
% x_k_sdp=sdpvar(123,1,'full');
% ops = sdpsettings('verbose',0,'solver','GUROBI');
%%
local_n=floor(size(A,1)/agent_num);
tic;
%% 算法主体
for k=1:Maxgen
    k   
    alpha=0.2;%0.25  0.01
    % 读取上次迭代的数据
    if k==1
      x_k_last=ones(123,agent_num);
    else
      x_k_last=x_k_store{k-1};
    end
%     if rem(k,5)==0
%         Ck=C{1,5};
%     else
%         Ck=C{1,rem(k,5)};
%     end
    Ck=C;
    % 开始循环算法
    for i=1:agent_num 
%        x_k_i_last=x_k_last(:,i);
        s_i=randperm(local_n,1);
       L_cut(i,:)=L_cut(i,:);
       A_cut(:,:,i)=A_cut(:,:,i);

        %-----求梯度--------
        mid=L_cut(i,:)'.*A_cut(:,:,i); 
        gradient=-mid(s_i,:).*exp(-mid(s_i,:)*x_k_last(:,i))./(1+exp(-mid(s_i,:)*x_k_last(:,i)));
        gradient=sum(gradient,1);%+2*lamuda2*x_k_i_last';
%         gradient_k(:,i)=gradient';
        clear mid;
        xx=zeros(123,1);
        for j=1:agent_num
            xx=xx+Ck(i,j)*x_k_last(:,j);
        end       
        x_k_i_new(:,i)=xx-alpha*gradient'-alpha*lamuda1*sign(x_k_last(:,i));

    end
     for i=1:agent_num
        
         q(:,i)= x_k_i_new(:,i);
         midd=L_cut(i,:)'.*A_cut(:,:,i);
        gg=sum(-midd.*exp(-midd*q(:,i))./(1+exp(-midd*q(:,i))),1)/local_n;
        gradient_k(:,i)=gg';
    end
    gradient_sto{k}=gradient_k;
     x_k_store{k}=x_k_i_new;%统一更新旧的待优化参数值
     
end

T=toc