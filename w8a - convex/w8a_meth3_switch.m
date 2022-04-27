clear all
%% 数据加载
load('w8a_smote.mat');%载入A：A(95598x300):95598个数据
load('L_w8a_smote.mat');%载入L：A(1x95598):95598个结果
load('data/C_meth1_smote_sw2_800.mat');%载入C_store
A=A1;
L=L1;
A=double(A);
L=double(L);
L(L==0)=-1;
L(L==1)=1;%由于正负样本比例是1:4
%% 参数设置
agent_num=10;% agent个数
Maxgen=200;% 迭代次数
C=C_store%默认对比算法与方法一的邻接矩阵是一样的
% C{1}=doubly_stochastic(10);% 生成5个随机的邻接矩阵（行和列和都是1）
% C{2}=doubly_stochastic(10);
% C{3}=doubly_stochastic(10);
% C{4}=doubly_stochastic(10);
% C{5}=doubly_stochastic(10);
% C_store=C;
tau_i=5;
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
% x_i_sdp=sdpvar(300,1,'full');
% ops = sdpsettings('verbose',0,'solver','GUROBI');
%% 初始化
x=ones(300,agent_num);% x阵
for i=1:agent_num
 %-----求梯度--------
 mid=L_cut(i,:)'.*A_cut(:,:,i);
  gradient_i_new=-mid.*exp(-mid*x(:,i))./(1+exp(-mid*x(:,i)));
%gradient_i_new=-mid.*exp(mid*x(:,i))./(1+exp(mid*x(:,i))).^2;
 gradient_i_new=sum(gradient_i_new,1)/floor(size(A,1)/agent_num);%+2*lamuda2*x(:,i)';
 clear mid;
 gradient(:,i)=gradient_i_new';
end
y=gradient;% y阵
pi=agent_num*y-gradient;% pai阵
z=zeros(300,agent_num);% z阵
tic;
%% 算法主体
for k=1:Maxgen
    k   
    % 开始循环算法
    for i=1:agent_num  
         %----------方法三：yalmip+gurobi求解--------------
    mid=L_cut(i,:)'.*A_cut(:,:,i);
     mid=log((1+exp(-mid*x(:,i))));
    % mid=-1./(1+exp(mid*x(:,i)));
     f=@(x_i_sdp)lamuda1*norm(x_i_sdp,1)+pi(:,i)'*(x_i_sdp-x(:,i))+sum(mid,1)/floor(size(A,1)/agent_num)+tau_i/2*norm(x_i_sdp-x(:,i),2)^2;
     clear mid;   
%      optimize([],f,ops);
     [xi,fval]=fminunc(f,x(:,i));
     xx(:,i)=value(xi);
        z(:,i)=x(:,i)+1/k^0.8*(xx(:,i)-x(:,i));
    end
    if rem(k,5)==0
        C_k=C{1,5};
    else
        C_k=C{1,rem(k,5)};
    end
    % C_k=lamda(C,k);%lameda(10x10)
    for i=1:agent_num     
        x_i_new=0;
        y_i_new=0;
        for j=1:agent_num
           x_i_new=x_i_new+C_k(i,j)*z(:,j);
           y_i_new=y_i_new+C_k(i,j)*y(:,j);
        end
        x(:,i)=x_i_new;
        %-----求梯度--------
        mid=L_cut(i,:)'.*A_cut(:,:,i); 
        gradient_i_new=-mid.*exp(-mid*x_i_new)./(1+exp(-mid*x_i_new));
        gradient_i_new=sum(gradient_i_new,1)/floor(size(A,1)/agent_num);%+2*lamuda2*x_i_new';
        clear mid;   
        y_i_new=y_i_new+gradient_i_new'-gradient(:,i);
        y(:,i)=y_i_new;
        pi(:,i)=agent_num*y_i_new-gradient_i_new';
        gradient(:,i)=gradient_i_new';
    end
    x_k_store{k}=x;
    gradient_sto{k}=gradient;
end   
T=toc

