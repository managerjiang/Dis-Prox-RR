clear all
%% 数据加载
load('a9a.mat');%载入A：A(32561x123):32561个数据
load('L_a9a.mat');%载入L：A(1x32561):32561个结果
A=double(A);
L=double(L);
L(L==0)=-1;
L(L==1)=1;
%% 参数设置
agent_num=10;% agent个数
Maxgen=500;% 迭代次数
C=doubly_stochastic(agent_num);% 生成随机的邻接矩阵（行和列和都是1）
x_k_i_last=zeros(123,agent_num);% 第k次迭代，每个智能体x的值
q=zeros(123,agent_num);% Q阵
alpha=0.4;% 步长
lamuda1=5*10^(-4);
lamuda2=10^(-4);
global v;% V阵

%% GUROBI设置
x_k_sdp=sdpvar(123,1,'full');
ops = sdpsettings('verbose',0,'solver','GUROBI');
%% 算法主体
for k=1:Maxgen
    k
    
    %每个智能体选择7841个正样本，以及随机的24720个数据中的7841个，每100次迭代更新一次
    if mod(k,2)==1
      for i=1:agent_num
      [A_cut(:,:,i),L_cut(i,:)]=combine_data(A,L);
      end
    end
    % 读取上次迭代的数据
    if k==1
      x_k_last=zeros(123,agent_num);
    else
      x_k_last=x_k_store{k-1};
    end
    % 开始循环算法
    for i=1:agent_num       
        x_k_i_last=x_k_last(:,i);
        %-----求梯度--------
        mid=L_cut(i,:)'.*A_cut(:,:,i); 
        gradient=-mid.*exp(mid*x_k_i_last)./(1+exp(mid*x_k_i_last)).^2;
        gradient=sum(gradient,1)/floor(size(A,1)/agent_num)+2*lamuda2*x_k_i_last';
        gradient_sto(:,k)=gradient';
        clear mid;
        %---------求q----------
        q(:,i)=x_k_i_last-alpha*gradient';%q(123x10)
        
    end
     %所有个体异步更新q值，再更新V和x
    for i=1:agent_num
         %---------求v-----------
          mid=0;%v(123x1)
        for j=1:agent_num
           mid=mid+C(i,j)*q(:,j);
        end
        v=mid;
        clear mid;
        %----------方法三：yalmip+gurobi求解--------------
     f=lamuda1*norm(x_k_sdp,1)+lamuda2*norm(x_k_sdp,2)^2+norm(x_k_sdp-v,2)^2/(2*alpha);
     reuslt = optimize([],f,ops);
     x_k_i_new(:,i)=value(x_k_sdp);
    end
     x_k_store{k}=x_k_i_new;%统一更新旧的待优化参数值
     
end

