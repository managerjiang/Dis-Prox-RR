close all;
clc;
clear;

fval =0.3864;
%% meth1_sw
% ==============需要读取.mat文件================
  load('data/G1_meth1_smote_sw_200.mat');
  load('data/X1_meth1_smote_sw_200.mat');
  load('data/G2_meth1_smote_sw_200.mat');%
  load('data/X2_meth1_smote_sw_200.mat');
  load('data/G3_meth1_smote_sw_200.mat');
  load('data/X3_meth1_smote_sw_200.mat');
  load('data/G4_meth1_smote_sw_200.mat');
  load('data/X4_meth1_smote_sw_200.mat');
  load('data/C_meth1_smote_sw_800.mat');%加载保存的邻接矩阵信息C_store
  lamuda1=5*10^(-4);
%   lamuda2=10^(-4);
  load('a9a.mat');
  load('L_a9a.mat');
  L=double(L);
  L(L==0)=-1;
  C=C_store{1};
% 参数设置
Maxgen1s=size(x_k_store,2);%迭代次数
agent_num=size(C);%智能体个数

% 训练集
for k=1:Maxgen1s
  x_k=x_k_store{k};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  % 目标函数
%   fi=sum(log((1+exp(-L'.*A*x_k_1))),1);
%   obj_m1s(k)=fi/size(A,1)+lamuda1*norm(x_k_1,1)-fval;%+lamuda2*norm(x_k_1,2)^2; 
  [fv1,gv1]=fungk(x_k_store{k}(:,1),gradient_sto{k}(:,1),A,L);
  [fv2,gv2]=fungk(x_k_store2{k}(:,1),gradient_sto2{k}(:,1),A,L);
  [fv3,gv3]=fungk(x_k_store3{k}(:,1),gradient_sto3{k}(:,1),A,L);
  [fv4,gv4]=fungk(x_k_store4{k}(:,1),gradient_sto4{k}(:,1),A,L);
  obj_m1s(k)=(fv1+fv2+fv3+fv4)/4;
  zz_m1s(k)=(gv1+gv2+gv3+gv4)/4;
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
%    g_k=gradient_sto{k}+lamuda1*sign(x_k_1);
%    zz_m1s(k)=norm(g_k(:,1));
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

%% meth 2
% ==============需要读取.mat文件================
  load('data/G_meth2_smote_sw_200.mat');%加载保存的迭代梯度信息x_k_store{}
  load('data/X_meth2_smote_sw_200.mat');%加载保存的迭代解信息gradient()
%   load('data/C_meth1_smote_sw2_800.mat');%加载保存的邻接矩阵信息C_store
  lamuda1=5*10^(-4);
%   lamuda2=10^(-4);
  load('a9a.mat');
  load('L_a9a.mat');
  L=double(L);
  L(L==0)=-1;
%   C=C_store{1};
% 参数设置
Maxgen2s=size(x_k_store,2);%迭代次数
agent_num=size(C);%智能体个数

% 训练集
for k=1:Maxgen2s
  x_k=x_k_store{k};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  % 目标函数
  %fi=sum(1./(1+exp(L'.*A*x_k_1)),1);
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
   g_k=gradient_sto{k}+lamuda1*sign(x_k_1);
   zz_m2s(k)=norm(g_k(:,1));
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
%% meth3
% ==============需要读取.mat文件================
  load('data/G_meth3_smote_sw_200.mat');%加载保存的迭代梯度信息x_k_store{}
  load('data/X_meth3_smote_sw_200.mat');%加载保存的迭代解信息gradient()
%   load('data/C_meth1_smote_sw2_800.mat');%加载保存的邻接矩阵信息C_store
  lamuda1=5*10^(-4);
  lamuda2=10^(-4);
  load('a9a.mat');
  load('L_a9a.mat');
  L=double(L);
  L(L==0)=-1;
%   C=C_store{1};
% 参数设置
Maxgen3s=size(x_k_store,2);%迭代次数
agent_num=size(C);%智能体个数

% 训练集
for k=1:Maxgen3s
  x_k=x_k_store{k};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  % 目标函数
  %fi=sum(1./(1+exp(L'.*A*x_k_1)),1);
  fi=sum(log((1+exp(-L'.*A*x_k_1))),1);
  obj_m3(k)=fi/size(A,1)+lamuda1*norm(x_k_1,1)-fval;%+lamuda2*norm(x_k_1,2)^2; 
  %  XLX
  XLX=0;
  for i=1:agent_num
    temp_X=zeros(size(A,2),1);
    for j=1:agent_num
        temp_X=temp_X+C(i,j)*(x_k(:,i)-x_k(:,j));
    end
    XLX=XLX+x_k(:,i)'*temp_X;
  end
  w_m3(k)=XLX;
  % 梯度值
   %取所有智能体中第一个
   g_k=gradient_sto{k}+lamuda1*sign(x_k_1);
   zz_m3(k)=norm(g_k(:,1));
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

  yy_m3(k)=sum((result==L'))/size(L,2);
  
end


%% meth4
% ==============需要读取.mat文件================
  load('data/G_meth4_smote_sw_200.mat');%加载保存的迭代梯度信息x_k_store{}
  load('data/X_meth4_smote_sw_200.mat');%加载保存的迭代解信息gradient()
%   load('data/C_meth1_smote_sw2_800.mat');%加载保存的邻接矩阵信息C_store
  lamuda1=5*10^(-4);
  lamuda2=5*10^(-4);
  load('a9a.mat');
  load('L_a9a.mat');
  L=double(L);
  L(L==0)=-1;
%   C=C_store{1};
% 参数设置
Maxgen4s=size(x_k_store,2);%迭代次数
agent_num=size(C);%智能体个数

% 训练集
for k=1:Maxgen4s
  x_k=x_k_store{k};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  % 目标函数
  fi=sum(log((1+exp(-L'.*A*x_k_1))),1);
  obj_m4(k)=fi/size(A,1)+lamuda1*norm(x_k_1,1)-fval;%+lamuda2*norm(x_k_1,2)^2; 
  %  XLX
  XLX=0;
  for i=1:agent_num
    temp_X=zeros(size(A,2),1);
    for j=1:agent_num
        temp_X=temp_X+C(i,j)*(x_k(:,i)-x_k(:,j));
    end
    XLX=XLX+x_k(:,i)'*temp_X;
  end
  w_m4(k)=XLX;
  % 梯度值
   %取所有智能体中第一个
   g_k=gradient_sto{k}+lamuda1*sign(x_k_1);
   zz_m4(k)=norm(g_k(:,1));
end
% 测试集
  load('a9a_test.mat');
  load('L_a9a_test.mat');
  L=double(L);
  L(L==0)=-1;
for k=1:Maxgen4s
     % 测试集正确率
  x_k=x_k_store{k};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  result=A*x_k_1;
  result(result>=0)=1;
  result(result<0)=-1;

  yy_m4(k)=sum((result==L'))/size(L,2);
  
end

%% 绘图  
dxnum=60;
figure(1);
plot(1:5:dxnum,w_m1s(1:5:dxnum),'r-v','linewidth',1),hold on;
plot(1:5:dxnum,w_m2s(1:5:dxnum),'b-x','linewidth',1),hold on;
plot(1:5:dxnum,w_m3(1:5:dxnum),'k-o','linewidth',1), hold on;
 plot(1:5:dxnum,w_m4(1:5:dxnum),'g-.','linewidth',1.3), hold on;
legend('DPG-RR','Prox-G','NEXT','DGM');
ylabel('$$\|D(x)\|$$','Interpreter','latex')
xlabel('iterations');
figure(2);
subplot(1,2,1)
plot(1:10:Maxgen1s,zz_m1s(1:10:Maxgen1s),'r-v','linewidth',1),hold on;
plot(1:10:Maxgen2s,zz_m2s(1:10:Maxgen2s),'b-x','linewidth',1),hold on;
plot(1:10:Maxgen3s,zz_m3(1:10:Maxgen3s),'k-o','linewidth',1), hold on;
 plot(1:10:Maxgen4s,zz_m4(1:10:Maxgen4s),'g-.','linewidth',1.3), hold on;
legend('DPG-RR','Prox-G','NEXT','DGM');
ylabel('$$\|\nabla F(x)\|$$','Interpreter','latex')
xlabel('iterations')

subplot(1,2,2)
plot(1:10:Maxgen1s,obj_m1s(1:10:Maxgen1s),'r-v','linewidth',1),hold on;
plot(1:10:Maxgen2s,obj_m2s(1:10:Maxgen2s),'b-x','linewidth',1),hold on;
plot(1:10:Maxgen3s,obj_m3(1:10:Maxgen3s),'k-o','linewidth',1), hold on;
 plot(1:10:Maxgen4s,obj_m4(1:10:Maxgen4s),'g-.','linewidth',1.3), hold on;
legend('DPG-RR','Prox-G','NEXT','DGM');
ylabel('$$F(x)-F_*$$','Interpreter','latex')
xlabel('iterations')
axes('Position',[0.65,0.32,0.22,0.18]);   % set the small figure     

plot(20:10:80,obj_m1s(20:10:80),'r-v','linewidth',1), hold on;      
plot(20:10:80,obj_m2s(20:10:80),'b-x','linewidth',1), hold on;% plot the local small figure                                                                                                               
xlim([20,80]);                         % set the axes range  


