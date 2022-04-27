function [fv,gkv]=fungk(xx,grad,A,L)
fval =0.3864;
  lamuda1=5*10^(-4);
    fv=0;
    gkv=0;
    fi=sum(log((1+exp(-L'.*A*xx))),1);
    fv=fi/size(A,1)+lamuda1*norm(xx,1)-fval;%+lamuda2*norm(x_k_1,2)^2;         
    g_k=grad+lamuda1*sign(xx);
    gkv=norm(g_k);
   
end