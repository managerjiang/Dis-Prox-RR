function [fv,gkv,XLX]=fungk(xstore,xx,grad,A,L,C)
fval =0.38;
  lamuda1=5*10^(-4);
    fv=0;
    gkv=0;
    fi=sum(log((1+exp(-L'.*A*xx))),1);
    fv=fi/size(A,1)+lamuda1*norm(xx,1)-fval;%+lamuda2*norm(x_k_1,2)^2;         
    g_k=grad+lamuda1*sign(xx);
    gkv=norm(g_k);
   XLX=0;
   agent_num=size(C,1);
  for i=1:agent_num
    temp_X=zeros(size(A,2),1);
    for j=1:agent_num
        temp_X=temp_X+C(i,j)*(xstore(:,i)-xstore(:,j));
    end
    XLX=XLX+xstore(:,i)'*temp_X;
  end
end