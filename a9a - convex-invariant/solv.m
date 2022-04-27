clear 

load('a9a_smote.mat');%载入A：A(48243x123):48243个数据
load('L_a9a_smote.mat');%载入L：A(1x48243):48243个结果
A=A1;
L=L1;
A=double(A);
L=double(L);
L(L==0)=-1;
L(L==1)=1;%由于正负样本比例是1:4
 lamuda1=5*10^(-4);
 lamuda2=5*10^(-4);
obj_m=@(x_k)sum(log((1+exp(-L'.*A*x_k))),1)/size(A,1)+lamuda1*norm(x_k,1); 
 [x,fval]=fminunc(obj_m,ones(123,1));
 x
 fval