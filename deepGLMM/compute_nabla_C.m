function f = compute_nabla_C(b,c)
% 

c2 = c.^2;
b2 = b.^2;
alpha = 1/(1+sum(b2./c2));
Cminus_b = b./c2;
Sigma_inv = diag(1./c2)-alpha*(Cminus_b*Cminus_b');

f = [zeros(size(b));-alpha*Cminus_b;-c.*diag(Sigma_inv)];

end

