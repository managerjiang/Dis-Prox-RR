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
% C{1}=doubly_stochastic(10);% 生成5个随机的邻接矩阵（行和列和都是1）
% C{2}=doubly_stochastic(10);
% C{3}=doubly_stochastic(10);
% C{4}=doubly_stochastic(10);
% C{5}=doubly_stochastic(10);
% C_store=C;
load('data/C_meth1_smote_sw2_800.mat');%载入C_store
C=C_store;
x_k_i_last=zeros(123,agent_num);% 第k次迭代，每个智能体x的值
q=zeros(123,agent_num);% Q阵
alpha=0.3;% 步长
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
x_k_sdp=sdpvar(123,1,'full');
ops = sdpsettings('verbose',0,'solver','GUROBI');
tic;
%% 算法主体
for k=1:Maxgen
    k   
%     alpha=1/k^0.2;
    % 读取上次迭代的数据
    if k==1
      x_k_last=ones(123,agent_num);
    else
      x_k_last=x_k_store{k-1};
    end
    % 开始循环算法
    local_n=floor(size(A,1)/agent_num);
    local_batch=300;
    for i=1:agent_num       
        x_k_i_last=x_k_last(:,i);
        %-----求梯度--------
        mid=L_cut(i,:)'.*A_cut(:,:,i); 
        gradient=-mid.*exp(-mid*x_k_i_last)./(1+exp(-mid*x_k_i_last));
        gradient=sum(gradient,1)/local_n;%+2*lamuda2*x_k_i_last';
        gradient_k(:,i)=gradient';
        clear mid;
        %---------求q----------
        q(:,i)=x_k_i_last-alpha*gradient';%q(123x10)
    end
    gradient_sto{k}=gradient_k;
     %所有个体异步更新q值，再更新V和x
     C_k=lamda(C,k);%lameda(10x10)
    for i=1:agent_num
         %---------求v-----------
          mid=0;%v(123x1)
        for j=1:agent_num
           mid=mid+C_k(i,j)*q(:,j);
        end
        v=mid;
        clear mid;
        %----------方法三：yalmip+gurobi求解--------------
        f=@(x_k_sdp)lamuda1*norm(x_k_sdp,1)+norm(x_k_sdp-v,2)^2/(2*alpha);
     [x,fval]=fminunc(f,x_k_i_last);
     x_k_i_new(:,i)=x;
    end
     x_k_store{k}=x_k_i_new;%统一更新旧的待优化参数值
     
end
T=toc
