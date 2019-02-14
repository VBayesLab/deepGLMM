function grad_h = grad_h_theta(theta,X,loglik,logprior)
% Caluclate grad of h(theta)
q = size(X,2); % Number of feature

[~, dg_lik] = loglik.name(theta(1:q), loglik.inargs{:});
[~, dg_prior] = logprior.name(theta, logprior.inargs{:},q);

dg_lik = [dg_lik ; zeros(q,1)];
grad_h = dg_lik + dg_prior;
end

