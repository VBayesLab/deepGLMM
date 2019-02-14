function [mu,Sigma,Zi] = newton_raphson_simplified(W_seq,theta_gammaj,Xi)
% The newton raphson procedure for computing the mode \hat{mu}_{alpha_i}
M = size(W_seq{end},1);
Zi = features_output(Xi,W_seq);
Sigma = diag(exp(theta_gammaj));
mu = zeros(M+1,1);
end

