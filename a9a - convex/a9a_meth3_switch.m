clear all
%% ���ݼ���
load('a9a_smote.mat');%����A��A(48243x123):48243������
load('L_a9a_smote.mat');%����L��A(1x48243):48243�����
load('data/C_meth1_smote_sw2_800.mat');%����C_store
A=A1;
L=L1;
A=double(A);
L=double(L);
L(L==0)=-1;
L(L==1)=1;%������������������1:4
%% ��������
agent_num=10;% agent����
Maxgen=200;% ��������
C=C_store%Ĭ�϶Ա��㷨�뷽��һ���ڽӾ�����һ����

tau_i=5;% 
lamuda1=0.5*10^(-5);
lamuda2=0.5*10^(-5);
global v;% V��
%% ����Ԥ����
%��������������ü����ݣ�ÿ��������ʮ��֮һ������
for i=1:agent_num
    L_cut(i,:)=L((i-1)*floor(size(A,1)/agent_num)+1:i*floor(size(A,1)/agent_num));
    A_cut(:,:,i)=A((i-1)*floor(size(A,1)/agent_num)+1:i*floor(size(A,1)/agent_num),:); 
end
%% GUROBI����
% x_i_sdp=sdpvar(123,1,'full');
% ops = sdpsettings('verbose',0,'solver','GUROBI');
%% ��ʼ��
x=ones(123,agent_num);% x��
for i=1:agent_num
 %-----���ݶ�--------
 mid=L_cut(i,:)'.*A_cut(:,:,i);
 gradient_i_new=-mid.*exp(-mid*x(:,i))./(1+exp(-mid*x(:,i)));
 %gradient_i_new=-mid.*exp(mid*x(:,i))./(1+exp(mid*x(:,i))).^2;
 gradient_i_new=sum(gradient_i_new,1)/floor(size(A,1)/agent_num);
 clear mid;
 gradient(:,i)=gradient_i_new';
end
y=gradient;% y��
pi=agent_num*y-gradient;% pai��
z=zeros(123,agent_num);% z��
tic;
%% �㷨����
for k=1:Maxgen
    k   
    % ��ʼѭ���㷨
    for i=1:agent_num  
         %----------��������yalmip+gurobi���--------------
     mid=L_cut(i,:)'.*A_cut(:,:,i);
     mid=log((1+exp(-mid*x(:,i))));
    % mid=-1./(1+exp(mid*x(:,i)));
     f=@(x_i_sdp)lamuda1*norm(x_i_sdp,1)+pi(:,i)'*(x_i_sdp-x(:,i))+sum(mid,1)/floor(size(A,1)/agent_num)+tau_i/2*norm(x_i_sdp-x(:,i),2)^2;
     clear mid;   
%      optimize([],f,ops);
     [xi,fval]=fminunc(f,x(:,i));
     xx(:,i)=xi;
        z(:,i)=x(:,i)+1/k*(xx(:,i)-x(:,i));
    end
    if rem(k,5)==0
        C_k=C{1,5};
    else
        C_k=C{1,rem(k,5)};
    end
  %   C_k=lamda(C,k);%lameda(10x10)
    for i=1:agent_num     
        x_i_new=0;
        y_i_new=0;
        for j=1:agent_num
           x_i_new=x_i_new+C_k(i,j)*z(:,j);
           y_i_new=y_i_new+C_k(i,j)*y(:,j);
        end
        x(:,i)=x_i_new;
        %-----���ݶ�--------
        mid=L_cut(i,:)'.*A_cut(:,:,i); 
      %  gradient_i_new=-mid.*exp(mid*x_i_new)./(1+exp(mid*x_i_new)).^2;
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

