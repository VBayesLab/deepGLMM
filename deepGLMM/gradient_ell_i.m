function f = gradient_ell_i(W_seq,beta,yi,Xi,theta_gammaj,N,d,distr,varargin)
% Compute estiamte of the gradient of the log-likelihood contribution
% ell_i(theta)
if strcmp(distr,'binomial')
    %[mu_alpha_i,Sigma_alpha_i,Zi] = newton_raphson(W1,W2,beta,theta_gammaj,yi,Xi);
    [mu_alpha_i,Sigma_alpha_i,Zi] = newton_raphson_simplified(W_seq,theta_gammaj,Xi);
    Gamma_inv = diag(exp(-theta_gammaj));
    % Now compute the gradient of the log-likelihood contribution \ell_i
    gradient_j = zeros(d,N);
    log_weights = zeros(N,1);
    length_alpha = length(mu_alpha_i);
    auxSigma = chol(Sigma_alpha_i,'lower');
    U_normal = normrnd_qmc(length_alpha,N);
    for jsample = 1:N    
        alpha_i = mu_alpha_i+auxSigma*U_normal(:,jsample);

        log_weights(jsample) = yi'*Zi*(beta+alpha_i)- sum(log(1+exp(Zi*(beta+alpha_i))))-1/2*alpha_i'*Gamma_inv*alpha_i+...
            +1/2*(alpha_i-mu_alpha_i)'*(Sigma_alpha_i\(alpha_i-mu_alpha_i));

        gradient_w_beta = nn_backpropagation(Xi,yi,W_seq,beta,alpha_i,distr);
        gradient_theta_gammaj = -1/2+1/2*exp(-theta_gammaj).*(alpha_i.^2);
        gradient_j(:,jsample) = [gradient_w_beta;gradient_theta_gammaj];
    end
elseif strcmp(distr,'normal')
    theta_sigma2 = varargin{1};
    sigma2 = exp(theta_sigma2);
    [mu_alpha_i,Sigma_alpha_i,Zi] = get_mean_var(W_seq,beta,theta_gammaj,sigma2,yi,Xi);
    Gamma_inv = diag(exp(-theta_gammaj));
    % Now compute the gradient of the log-likelihood contribution \ell_i
    gradient_j = zeros(d,N);
    log_weights = zeros(N,1);
    Ti = length(yi);
    length_alpha = length(mu_alpha_i);
    auxSigma = chol(Sigma_alpha_i,'lower');
    U_normal = normrnd_qmc(length_alpha,N);
    for jsample = 1:N    
        alpha_i = mu_alpha_i+auxSigma*U_normal(:,jsample);
        node_store = neural_net_output(Xi,W_seq,beta,alpha_i);
        nn_output = node_store{end}; % Neuron net output

        Mi = (yi-Zi*beta)'*(Zi*alpha_i)-0.5*alpha_i'*(Zi'*Zi)*alpha_i;
        f_alpha_i = sigma2^-1*Mi - 0.5*alpha_i'*Gamma_inv*alpha_i;

        log_weights(jsample) = f_alpha_i+...
            +1/2*(alpha_i-mu_alpha_i)'*(Sigma_alpha_i\(alpha_i-mu_alpha_i));

        gradient_w_beta = exp(-theta_sigma2)*nn_backpropagation(Xi,yi,W_seq,beta,alpha_i,distr);

        gradient_theta_gammaj = -1/2+1/2*exp(-theta_gammaj).*(alpha_i.^2);
        %-----------------------------------------------------------------------------
        gradient_theta_sigma2 = -1/2*Ti+1/2*exp(-theta_sigma2)*sum((yi-nn_output').^2);
        %-----------------------------------------------------------------------------
        gradient_j(:,jsample) = [gradient_w_beta;gradient_theta_gammaj;gradient_theta_sigma2];
    end
else
    disp('Distribution must be either Normal or Binomial')
end
log_weights = log_weights-max(log_weights);
Weights = exp(log_weights)/sum(exp(log_weights));
f = gradient_j*Weights;

end



    
