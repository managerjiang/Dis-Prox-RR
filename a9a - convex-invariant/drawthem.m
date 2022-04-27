close all;
clc;
fval =0.38;
%%meth1
% ==============需要读取.mat文件================
  load('data/G_meth1_smote_sw_800.mat');%加载保存的迭代梯度信息x_k_store{}
  load('data/X_meth1_smote_sw_800.mat');%加载保存的迭代解信息gradient()
  load('data/C_meth1_smote_800.mat');%加载保存的邻接矩阵信息C_store

lamuda1=5*10^(-4);
%   lamuda2=10^(-4);
  load('a9a.mat');
  load('L_a9a.mat');
  L=double(L);
  L(L==0)=-1;
  C=C_store;
% 参数设置
Maxgen1s=size(x_k_store,2);%迭代次数
agent_num=size(C,1);%智能体个数

% 训练集
for k=1:Maxgen1s
  x_k=x_k_store{k};
  x_k_1=sum(x_k,2)./agent_num;%取所有智能体中第一个
  % 目标函数
  fi=sum(log((1+exp(-L'.*A*x_k_1))),1);
  obj_m1s(k)=fi/size(A,1)+lamuda1*norm(x_k_1,1)-fval;%+lamuda2*norm(x_k_1,2)^2; 
  %  XLX
  XLX=0;
  for i=1:agent_num
    temp_X=zeros(size(A,2),1);
    for j=1:agent_num
        temp_X=temp_X+C(i,j)*(x_k(:,i)-x_k(:,j));
    end
    XLX=XLX+x_k(:,i)'*temp_X;
  end
  w_m1s(k)=XLX;
  % 梯度值
   %取所有智能体中第一个
   g_k=sum(gradient_sto{k},2)./agent_num+lamuda1*sign(x_k_1);
   zz_m1s(k)=norm(g_k);
end
% 测试集
  load('a9a_test.mat');
  load('L_a9a_test.mat');
  L=double(L);
  L(L==0)=-1;
for k=1:Maxgen1s
     % 测试集正确率
  x_k=x_k_store{k};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  result=A*x_k_1;
  result(result>=0)=1;
  result(result<0)=-1;

  yy_m1s(k)=sum((result==L'))/size(L,2);
end
%%meth2
% ==============需要读取.mat文件================
  load('data/G_meth2_smote_sw_800.mat');%加载保存的迭代梯度信息x_k_store{}
  load('data/X_meth2_smote_sw_800.mat');%加载保存的迭代解信息gradient()
%   load('data/C_meth1_smote_800.mat');%加载保存的邻接矩阵信息C_store

lamuda1=5*10^(-4);
%   lamuda2=10^(-4);
  load('a9a.mat');
  load('L_a9a.mat');
  L=double(L);
  L(L==0)=-1;
%   C=C_store{1};
% 参数设置
Maxgen2s=size(x_k_store,2);%迭代次数
agent_num=size(C,1);%智能体个数

% 训练集
for k=1:Maxgen2s
  x_k=x_k_store{k};
  x_k_1=sum(x_k,2)./agent_num;
%   x_k_1=x_k(:,1);%取所有智能体中第一个
  % 目标函数
  fi=sum(log((1+exp(-L'.*A*x_k_1))),1);
  obj_m2s(k)=fi/size(A,1)+lamuda1*norm(x_k_1,1)-fval;%+lamuda2*norm(x_k_1,2)^2; 
  %  XLX
  XLX=0;
  for i=1:agent_num
    temp_X=zeros(size(A,2),1);
    for j=1:agent_num
        temp_X=temp_X+C(i,j)*(x_k(:,i)-x_k(:,j));
    end
    XLX=XLX+x_k(:,i)'*temp_X;
  end
  w_m2s(k)=XLX;
  % 梯度值
   %取所有智能体中第一个
   g_k=sum(gradient_sto{k},2)/agent_num+lamuda1*sign(x_k_1);
   zz_m2s(k)=norm(g_k);
end
% 测试集
  load('a9a_test.mat');
  load('L_a9a_test.mat');
  L=double(L);
  L(L==0)=-1;
for k=1:Maxgen2s
     % 测试集正确率
  x_k=x_k_store{k};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  result=A*x_k_1;
  result(result>=0)=1;
  result(result<0)=-1;

  yy_m2s(k)=sum((result==L'))/size(L,2);
end

%%meth3
% ==============需要读取.mat文件================
  load('data/G_meth3_smote_sw_800.mat');%加载保存的迭代梯度信息x_k_store{}
  load('data/X_meth3_smote_sw_800.mat');%加载保存的迭代解信息gradient()
%   load('data/C_meth1_smote_sw_800.mat');%加载保存的邻接矩阵信息C_store

lamuda1=5*10^(-4);
%   lamuda2=10^(-4);
  load('a9a.mat');
  load('L_a9a.mat');
  L=double(L);
  L(L==0)=-1;
%   C=C_store{1};
% 参数设置
Maxgen3s=size(x_k_store,2);%迭代次数
agent_num=size(C,1);%智能体个数

% 训练集
for k=1:Maxgen3s
  x_k=x_k_store{k};
    x_k_1=sum(x_k,2)./agent_num;
%   x_k_1=x_k(:,1);%取所有智能体中第一个
  % 目标函数
  fi=sum(log((1+exp(-L'.*A*x_k_1))),1);
  obj_m3s(k)=fi/size(A,1)+lamuda1*norm(x_k_1,1)-fval;%+lamuda2*norm(x_k_1,2)^2; 
  %  XLX
  XLX=0;
  for i=1:agent_num
    temp_X=zeros(size(A,2),1);
    for j=1:agent_num
        temp_X=temp_X+C(i,j)*(x_k(:,i)-x_k(:,j));
    end
    XLX=XLX+x_k(:,i)'*temp_X;
  end
  w_m3s(k)=XLX;
  % 梯度值
   %取所有智能体中第一个
   g_k=sum(gradient_sto{k},2)/agent_num+lamuda1*sign(x_k_1);
   zz_m3s(k)=norm(g_k);
end
% 测试集
  load('a9a_test.mat');
  load('L_a9a_test.mat');
  L=double(L);
  L(L==0)=-1;
for k=1:Maxgen3s
     % 测试集正确率
  x_k=x_k_store{k};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  result=A*x_k_1;
  result(result>=0)=1;
  result(result<0)=-1;

  yy_m3s(k)=sum((result==L'))/size(L,2);
end


%% 绘图  
figure(1);
plot(1:10:Maxgen1s,w_m1s(1:10:Maxgen1s),'r-v','linewidth',1),hold on;
plot(1:10:Maxgen2s,w_m2s(1:10:Maxgen2s),'b-o','linewidth',1),hold on;
plot(1:10:Maxgen3s,w_m3s(1:10:Maxgen3s),'g-x','linewidth',1),hold on;
ylabel('$$\|D(x)\|$$','Interpreter','latex')
xlabel('iterations');
legend('DPG-RR','GT-SAGA','DSGD');
figure(2);
subplot(1,2,1)
plot(1:10:Maxgen1s,zz_m1s(1:10:Maxgen1s),'r-v','linewidth',1),hold on;
plot(1:10:Maxgen2s,zz_m2s(1:10:Maxgen2s),'b-o','linewidth',1),hold on;
plot(1:10:Maxgen3s,zz_m3s(1:10:Maxgen3s),'g-x','linewidth',1),hold on;
ylabel('$$\|\nabla F(x)\|$$','Interpreter','latex')
xlabel('iterations')
legend('DPG-RR','GT-SAGA','DSGD');
subplot(1,2,2)
plot(1:10:Maxgen1s,obj_m1s(1:10:Maxgen1s),'r-v','linewidth',1),hold on;
plot(1:10:Maxgen2s,obj_m2s(1:10:Maxgen2s),'b-o','linewidth',1),hold on;
plot(1:10:Maxgen3s,obj_m3s(1:10:Maxgen3s),'g-x','linewidth',1),hold on;
ylabel('$$F(x)-F_*$$','Interpreter','latex')
xlabel('iterations')
legend('DPG-RR','GT-SAGA','DSGD');
 
