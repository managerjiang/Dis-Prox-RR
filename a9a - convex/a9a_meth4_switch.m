clear all
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

load('data/C_meth1_smote_sw2_800.mat');%载入C_store
C=C_store;
x_k_i_last=zeros(123,agent_num);% 第k次迭代，每个智能体x的值
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
    alpha=1/k;
    % 读取上次迭代的数据
    if k==1
      x_k_last=ones(123,agent_num);
    else
      x_k_last=x_k_store{k-1};
    end
    if rem(k,5)==0
        Ck=C{1,5};
    else
        Ck=C{1,rem(k,5)};
    end
    % 开始循环算法
    local_n=floor(size(A,1)/agent_num);
    local_batch=local_n;
    for i=1:agent_num 
       x_k_i_last=x_k_last(:,i);
        %-----求梯度--------
        mid=L_cut(i,:)'.*A_cut(:,:,i); 
        gradient=-mid.*exp(-mid*x_k_i_last)./(1+exp(-mid*x_k_i_last));
        gradient=sum(gradient,1)/floor(size(A,1)/agent_num);%+2*lamuda2*x_k_i_last';
        gradient_k(:,i)=gradient';
        clear mid;
        %%----求合并信息
        x_consens=zeros(123,1);
        for j=1:agent_num
            x_consens=x_consens+Ck(i,j)*x_k_last(:,j);
        end
        %---------求q----------
        x_k_i_new(:,i)=x_consens-alpha*gradient'-alpha*lamuda1*sign(x_k_i_last);%q(123x10)
    end
    gradient_sto{k}=gradient_k;
     x_k_store{k}=x_k_i_new;%统一更新旧的待优化参数值
     
end
T=toc
