function prod = inverse_fisher_times_grad(b,c,gradient_bar)
% compute the product inverse_fisher x grad

d = length(b);
grad1 = gradient_bar(1:d);
grad2 = gradient_bar(d+1:2*d);
grad3 = gradient_bar(2*d+1:end);
%grad2 = gradient_bar(d+1:end);

c2 = c.^2;
b2 = b.^2;

prod1 = (b'*grad1)*b+(grad1.*c2);

alpha = 1/(1+sum(b2./c2));
Cminus = diag(1./c2);
Cminus_b = b./c2;
Sigma_inv = Cminus-alpha*(Cminus_b*Cminus_b');

%A11 = (1-alpha)*Cminus+(alpha^2-alpha*(1-alpha))*(Cminus_b*Cminus_b');
A11_inv = (1/(1-alpha))*((1-1/(sum(b2)+1-alpha))*(b*b')+diag(c2));

C = diag(c);
A12 = 2*(C*Sigma_inv*b*ones(1,d)).*Sigma_inv;
A21 = A12';
%h = c.*diag(Sigma_inv);
%A22 = 2*diag(h.^2);
A22 = 2*C*(Sigma_inv.*Sigma_inv)*C;

% W = A11\A12;
% C1 = A11-A12*(A22\A21);
% C2 = A22-A21*W;
% 
% prod2 = C1\grad2-W*(C2\grad3);
% prod3 = C2\(-W'*grad2+grad3);
%D = A22-A21*(A11\A12);
D = A22-A21*A11_inv*A12;
%prod2 = A11\grad2+(A11\A12)*(D\A21)*(A11\grad2)-(A11\A12)*(D\grad3);
prod2 = A11_inv*grad2+(A11_inv*A12)*(D\A21)*(A11_inv*grad2)-(A11_inv*A12)*(D\grad3);
%prod3 = -(D\A21)*(A11\grad2)+D\grad3;
prod3 = -(D\A21)*(A11_inv*grad2)+D\grad3;
%second_block = [A,U;U',W];
%prod2 = second_block\grad2;

prod = [prod1;prod2;prod3];
end

