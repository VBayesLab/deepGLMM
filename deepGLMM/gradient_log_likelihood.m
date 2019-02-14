function f = gradient_log_likelihood(W_seq,beta,y,X,theta_gammaj,d,distr,varargin)
% compute estimate of the gradient of the log-likelihood

n = length(y);
N = 20; % number of samples

gradient = 0;
for i = 1:n
    yi = y{i};
    Xi = X{i};
    if strcmp(distr,'normal')
        gradient_i = gradient_ell_i(W_seq,beta,yi,Xi,theta_gammaj,N,d,distr,varargin{1});
    else
        gradient_i = gradient_ell_i(W_seq,beta,yi,Xi,theta_gammaj,N,d,distr);
    end
    gradient = gradient+gradient_i;
end
f = gradient;

end



    
