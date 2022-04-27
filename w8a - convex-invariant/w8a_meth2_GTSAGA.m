clear all
%%GT-SAGA
%% 数据加载
load('w8a_smote.mat');%载入A：A(48243x123):48243个数据
load('L_w8a_smote.mat');%载入L：A(1x48243):48243个结果
A=A1;
L=L1;
A=double(A);
L=double(L);
L(L==0)=-1;
L(L==1)=1;%由于正负样本比例是1:4
%% 参数设置
agent_num=10;% agent个数
Maxgen=200;% 迭代次数
% C=doubly_stochastic(agent_num);
% C=doubly_stochastic(agent_num);% 生成随机的邻接矩阵（行和列和都是1）
% C_store=C
 load('data/C_meth1_smote_800.mat');%载入C_store
 C=C_store
%x_k_last=zeros(123,agent_num);% x阵
%  load('data/x_k_last_m.mat');
x_k_last=ones(300,agent_num);
% y_k_last=zeros(123,agent_num);% y阵
% v_k_last=zeros(123,agent_num);% v阵
lamuda1=0.5*10^(-5);
lamuda2=0.5*10^(-5);
agent_m=floor(size(A,1)/agent_num);
%% 数据预处理
%根据智能体个数裁剪数据，每个智能体十分之一的数据,同时变量初始化
for i=1:agent_num
    L_cut(:,i)=L((i-1)*floor(size(A,1)/agent_num)+1:i*floor(size(A,1)/agent_num));
    A_cut(:,:,i)=A((i-1)*floor(size(A,1)/agent_num)+1:i*floor(size(A,1)/agent_num),:); 
    mid=L_cut(:,i).*A_cut(:,:,i); 
   % gradient=-mid.*exp(mid*x_k_last(:,i))./(1+exp(mid*x_k_last(:,i))).^2;
    gradient=-mid.*exp(-mid*x_k_last(:,i))./(1+exp(-mid*x_k_last(:,i)));
    gradient=sum(gradient,1)/floor(size(A,1)/agent_num)+lamuda1*sign(x_k_last(:,i)');%+2*lamuda2*x_k_last(:,i)';
    y_k_last(:,i)=gradient';
    v_k_last(:,i)=gradient';
end
for i=1:agent_num
    for j=1:agent_m
        tau_k_last(:,i,j)=x_k_last(:,i);
    end
end
eta=0.2;%%步长
tic;
%% 算法主体
for k=1:Maxgen
    k   
	for i=1:agent_num
       % x_k_i_last=x_k_last(:,i);
        %-----更新x_i(k+1)--------
        xk=zeros(300,1);
        for j=1:agent_num
            xk=xk+C(i,j)*x_k_last(:,j);
        end
        x_k_new(:,i)=xk-eta*y_k_last(:,i);
    end
	x_k_last=x_k_new;
    for i=1:agent_num  
         mid=L_cut(:,i).*A_cut(:,:,i);  
        xk=x_k_last(:,i);
        s_i=randperm(agent_m,1);
        mid=L_cut(:,i).*A_cut(:,:,i); 
%         gradient_x=-mid(s_i,:)*exp(mid(s_i,:)*xk)/(1+exp(mid(s_i,:)*xk))^2;
        gradient_x=-mid(s_i,:)*exp(-mid(s_i,:)*xk)/(1+exp(-mid(s_i,:)*xk));
        gradient_x=sum(gradient_x,1)+lamuda1*sign(xk');%+2*lamuda2*xk';
        gradient_x_i_k=gradient_x';
        gradient_tau=-mid(s_i,:)*exp(-mid(s_i,:)*tau_k_last(:,i,s_i))/(1+exp(-mid(s_i,:)*tau_k_last(:,i,s_i)));
        gradient_tau=sum(gradient_tau,1)+lamuda1*sign(tau_k_last(:,i,s_i)');%+2*lamuda2*tau_k_last(:,i,s_i)';
        gradient_tau_i_k=gradient_tau';
		gradient_tau_sum=zeros(size(A,2),1);
         for j=1:agent_m
            gradient=-mid(j,:)*exp(-mid(j,:)*tau_k_last(:,i,j))/(1+exp(-mid(j,:)*tau_k_last(:,i,j)));
            gradient=sum(gradient,1)+lamuda1*sign(tau_k_last(:,i,j)');%2*lamuda2*tau_k_last(:,i,j)';
            gradient_tau_sum=gradient_tau_sum+gradient';
			if(j==s_i)
            tau_k_last(:,i,j)=xk;
			end
         end
        gradient_tau_sum=gradient_tau_sum/agent_m;
        v_k_i_new=gradient_x_i_k-gradient_tau_i_k+gradient_tau_sum;
        v_k_new(:,i)=v_k_i_new; 
		%---------更新y_i(k+1)----------
		ymid=zeros(300,1);%v(123x1)
        for j=1:agent_num
           ymid=ymid+C(i,j)*y_k_last(:,j);
        end
        y_k_new(:,i)=ymid+v_k_new(:,i)-v_k_last(:,i);
		
    end
     gradient_ite_sum=zeros(size(A,2),1);
    for i=1:agent_num
     mid=L_cut(:,i).*A_cut(:,:,i); 
    gradient_ite=-mid.*exp(-mid*x_k_new(:,i))./(1+exp(-mid*x_k_new(:,i)));
    gradient_ite=sum(gradient_ite,1)/floor(size(A,1)/agent_num);%+2*lamuda2*x_k_new(:,i)';
    gradient_ite_sum=gradient_ite_sum+gradient_ite';
    end
    gradient_sto{k}=gradient_ite_sum;
    x_k_store{k}=x_k_new;
    y_k_last=y_k_new;
    v_k_last=v_k_new;
    
     
end
T=toc
