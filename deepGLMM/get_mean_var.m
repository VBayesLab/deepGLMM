function [mu,Sigma,Zi] = get_mean_var( W_seq,beta,theta_gammaj,sigma2,yi,Xi )

M = size(W_seq{end},1);
Ti = size(yi,1);
Zi = zeros(Ti,M+1);
for t=1:Ti
    x_it = Xi(t,:)';
    z_it = features_output(x_it',W_seq);
    Zi(t,:) = z_it';
end
Gamma_inv = diag(exp(-theta_gammaj));
A = sigma2^-1*(Zi'*Zi) + Gamma_inv;
b = (sigma2^-1*(yi-Zi*beta)'*Zi)';
A_inv = A^-1;
mu = A_inv*b;
Sigma = A_inv;
end

