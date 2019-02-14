function [mu,Sigma,Zi] = newton_raphson(W_seq,beta,theta_gammaj,yi,Xi,distr,varargin)
% The newton raphson procedure for computing the mode \hat{mu}_{alpha_i}

M = size(W_seq{end},1);
Ti = size(yi,1);
Zi = zeros(Ti,M+1);
for t=1:Ti
    x_it = Xi(t,:)';
    z_it = features_output(x_it',W_seq);
    Zi(t,:) = z_it';
end
Gamma_inv = diag(exp(-theta_gammaj));
alpha_i = zeros(M+1,1);
iter = 0;
stop = false;

if ~isempty(varargin)
    sigma2 = varargin{1};
end

while ~stop
    iter = iter+1;
    alpha_i_old = alpha_i;
    
    if strcmp(distr,'binomial')
        pi = 1./(1+exp(-Zi*(beta+alpha_i)));
        gradient = Zi'*(yi-pi)-Gamma_inv*alpha_i;
        minusHessian = Zi'*diag(pi.*(1-pi))*Zi+Gamma_inv;
    elseif strcmp(distr,'normal')
        gradient = sigma2^-1*(Zi'*yi-(Zi'*Zi)*(beta+alpha_i))-Gamma_inv*alpha_i;
        minusHessian = sigma2^-1*(Zi'*Zi)+Gamma_inv;
    else
        disp('Distribution must be either Normal or Binomial')
    end
    alpha_i = alpha_i+minusHessian\gradient;
    if (norm(alpha_i-alpha_i_old)<0.01)||(iter>10)
        stop = true; 
    end
end
mu = alpha_i;
Sigma = minusHessian\eye(M+1);

end

