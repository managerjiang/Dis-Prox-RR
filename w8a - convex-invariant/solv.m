clear 
load('w8a_smote.mat');
load('L_w8a_smote.mat');
A=A1;
L=L1;
A=double(A);
L=double(L);
L(L==0)=-1;
L(L==1)=1;%由于正负样本比例是1:4
L=double(L);
L(L==0)=-1;
 lamuda1=5*10^(-4);
 lamuda2=5*10^(-4);
obj_m=@(x_k)sum(log((1+exp(-L'.*A*x_k))),1)/size(A,1)+lamuda1*norm(x_k,1); 
 [x,fval]=fminunc(obj_m,ones(300,1));
 x
 fval