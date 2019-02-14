function est = deepGLMMTrain(X,y,X_val,y_val,est)
% DEEPGLMMTRAIN Traing a deepGLMM model with continuous/binomial reponse y.
% Inverse gamma prior is used for sigma2
% INPUT
%   X_train, y_train:           Training data
%   X_validation, y_validation: Validation data
%   mdl:                        Model setting
% OUTPUT
%   est:                        Training results 

%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%   
%   https://github.com/VBayesLab/deepGLMM
%
%   Version: 1.0
%   LAST UPDATE: Feb, 2019

% Extract training data and settings from input struct
n_units = est.network;
lrate = est.lrate;
S = est.S;                       % Number of Monte Carlo samples to estimate the gradient
tau = est.tau;                   % Threshold before reducing constant learning rate eps0
grad_weight = est.momentum;      % Weight in the momentum 
cScale = est.cvar;               % Random scale factor to initialize b,c
patience = est.patience;         % Stop if test error not improved after patience_parameter iterations
max_iter = est.maxiter;          % Number of times learning algorithm scan entire training data
distr = est.dist;
seed = est.seed;
gamma_w = est.gamma_w; 
gamma_beta = est.gamma_beta; 
hp_beta_0 = est.hp_beta_0;

% Training parameters
iter = 1;                        % Current training iteration
idxPatience = 0;                 % Patient index for early stopping
stop = false;                    % Reset stop flag

% Set random seed if specified
if(~isnan(seed))
    rng(seed)
end

p = size(X{1},2)-1;              % Number of predictors

if(strcmp(est.initialize,'adaptive'))
    init = true;
else
    init = false;
end

L = length(n_units);             % The number of hidden layers
W_seq = cell(1,L);               % Sequence of weight matrices
index_track = zeros(1,L);        % Keep track of indices of Wj matrices: index_track(1) is the total elements in W1, index_track(2) is the total elements in W1 & W2,...
index_track(1) = n_units(1)*(p+1);
d_w = n_units(1)*(p+1);          % Total weights up to (and including) the last layer
w_tilde_index = n_units(1)+1:n_units(1)*(p+1); % Indices of non-biase weights, for regulization prior
for j = 2:L
    d_w = d_w+n_units(j)*(n_units(j-1)+1);
    index_track(j) = index_track(j-1)+n_units(j)*(n_units(j-1)+1);
    w_tilde_index = [w_tilde_index,(index_track(j-1)+n_units(j)+1):index_track(j)];
end
w_tilde_index = w_tilde_index';

d_beta = n_units(L)+1;           % Dimension of the weights beta connecting the last layer to the output
d_gamma = n_units(L)+1;          % For diagonal elements in latent covariance Gamma

if(strcmp(distr,'normal'))
    d = d_w+d_beta+d_gamma+1;    % Last parameter is for error variance
else
    d = d_w+d_beta+d_gamma;
end

% Initialize network
if(init==true)
    layers = [p+1 n_units 1];
    weights = InitializeNN(layers);
    mu=[];
    for i=1:length(layers)-1
        mu=[mu;weights{i}(:)];
    end
    if(strcmp(distr,'normal'))
        mu = [mu;normrnd(0,cScale,d_gamma+1,1)];
    else
        mu = [mu;normrnd(0,cScale,d_gamma,1)];
    end
else
    mu = normrnd(0,cScale,d,1);
end

b = normrnd(0,cScale,d,1);
c = cScale*ones(d,1);
lambda=[mu;b;c];

% Extract indexes for training 
W1 = reshape(mu(1:index_track(1)),n_units(1),p+1);
W_seq{1} = W1; 
for j = 2:L
    index = index_track(j-1)+1:index_track(j);
    Wj = reshape(mu(index),n_units(j),n_units(j-1)+1);
    W_seq{j} = Wj; 
end

beta = mu(d_w+1:d_w+d_beta);
beta_tilde_index = (d_w+2:d_w+d_beta)'; % index of beta without bias

%%  Calculate loss with initial variational parameters
if(strcmp(distr,'normal'))
    sigma2 = exp(mu(end)); % for error variance
    theta_gammaj = mu(d_w+d_beta+1:end-1);
    loss =  prediction_loss(W_seq,beta,theta_gammaj,y_val,X_val,y,X,distr,sigma2);
    Loss_DL(iter) = loss.pps;
    % Display current training results
    disp(['Iteration: ',num2str(iter),'   -  MSE: ',num2str(Loss_DL(iter))]);
elseif (strcmp(distr,'binomial'))
    theta_gammaj = mu(d_w+d_beta+1:end);
    loss =  prediction_loss(W_seq,beta,theta_gammaj,y_val,X_val,y,X,distr);
    Loss_DL(iter) = loss.pps;
    % Display current training results
    disp(['Iteration: ',num2str(iter),'   -  PPS: ',num2str(Loss_DL(iter))]);
else
end

%%  First iteration
disp('---------- Start Training Phase ----------')
grad_g_lik_store = zeros(S,3*d);
rqmc = normrnd_qmc(S,d+1);      
for s=1:S
    U_normal = rqmc(s,:)';
    epsilon1 = U_normal(1);
    epsilon2 = U_normal(2:end);
    theta = mu+b*epsilon1+c.*epsilon2;   
    
    W_seq = cell(1,L);        
    W1 = reshape(theta(1:index_track(1)),n_units(1),p+1);
    W_seq{1} = W1;
    W1_tilde = W1(:,2:end); % weights without biases                
    grad_prior_w = [zeros(n_units(1),1);-gamma_w*W1_tilde(:)];        
    for j = 2:L
        index = index_track(j-1)+1:index_track(j);
        Wj = reshape(theta(index),n_units(j),n_units(j-1)+1);
        W_seq{j} = Wj; 
        Wj_tilde = Wj(:,2:end);
        grad_prior_Wj = [zeros(n_units(j),1);-gamma_w*Wj_tilde(:)];        
        grad_prior_w = [grad_prior_w;grad_prior_Wj];
    end   
    beta = theta(d_w+1:d_w+d_beta); 
    beta_tilde = beta(2:end); % vector beta without intercept
    if(strcmp(distr,'normal'))
        theta_gammaj = theta(d_w+d_beta+1:end-1);
        theta_sigma2 = theta(end);
        grad_llh = gradient_log_likelihood(W_seq,beta,y,X,theta_gammaj,d,distr,theta_sigma2);
        grad_prior_theta_sigma2 = -1;
        grad_jacobian_normal = 1;
    else
        theta_gammaj = theta(d_w+d_beta+1:end);
        grad_llh = gradient_log_likelihood(W_seq,beta,y,X,theta_gammaj,d,distr);
        grad_prior_theta_sigma2 = [];
        grad_jacobian_normal = [];
    end
    
    grad_jacobian = [zeros(d_w+d_beta,1);ones(d_gamma,1);grad_jacobian_normal];       
    grad_prior_beta = [0;-gamma_beta*beta_tilde];
    grad_prior_theta_gammaj = -hp_beta_0*exp(theta_gammaj);
    grad_prior = [grad_prior_w;grad_prior_beta;grad_prior_theta_gammaj;grad_prior_theta_sigma2];
    grad_h = grad_jacobian+grad_prior+grad_llh;
    grad_g = [eye(d);epsilon1*eye(d);diag(epsilon2)];
    grad_g_lik_store(s,:)= (grad_g*grad_h)';       
end    
nabla_C = compute_nabla_C(b,c);
nabla_D = (mean(grad_g_lik_store))';
grad_lb = nabla_D-nabla_C;
gradient_lambda = inverse_fisher_times_grad(b,c,grad_lb);

norm_gradient = norm(gradient_lambda);
norm_gradient_seq1 = norm_gradient;
gradient_bar = gradient_lambda;

lambda_best = lambda;                           % Store current best variational parameters
idxPatience = 0;
while ~stop  
    iter = iter+1;
    rqmc = normrnd_qmc(S,d+1);      
    for s=1:S
        U_normal = rqmc(s,:)';
        epsilon1 = U_normal(1);
        epsilon2 = U_normal(2:end);
        theta = mu+b*epsilon1+c.*epsilon2;   
        
        W_seq = cell(1,L);        
        W1 = reshape(theta(1:index_track(1)),n_units(1),p+1);
        W_seq{1} = W1;
        W1_tilde = W1(:,2:end); % weights without biases                
        grad_prior_w = [zeros(n_units(1),1);-gamma_w*W1_tilde(:)];        
        for j = 2:L
            index = index_track(j-1)+1:index_track(j);
            Wj = reshape(theta(index),n_units(j),n_units(j-1)+1);
            W_seq{j} = Wj; 
            Wj_tilde = Wj(:,2:end);
            grad_prior_Wj = [zeros(n_units(j),1);-gamma_w*Wj_tilde(:)];        
            grad_prior_w = [grad_prior_w;grad_prior_Wj];
        end 
        beta = theta(d_w+1:d_w+d_beta); 
        beta_tilde = beta(2:end); % vector beta without intercept
        if(strcmp(distr,'normal'))
            theta_gammaj = theta(d_w+d_beta+1:end-1);
            theta_sigma2 = theta(end);
            grad_llh = gradient_log_likelihood(W_seq,beta,y,X,theta_gammaj,d,distr,theta_sigma2);
            grad_prior_theta_sigma2 = -1;
            grad_jacobian_normal = 1;
        else
            theta_gammaj = theta(d_w+d_beta+1:end);
            grad_llh = gradient_log_likelihood(W_seq,beta,y,X,theta_gammaj,d,distr);
            grad_prior_theta_sigma2 = [];
            grad_jacobian_normal = [];
        end

        grad_jacobian = [zeros(d_w+d_beta,1);ones(d_gamma,1);grad_jacobian_normal];       
        grad_prior_beta = [0;-gamma_beta*beta_tilde];
        grad_prior_theta_gammaj = -hp_beta_0*exp(theta_gammaj);
        grad_prior = [grad_prior_w;grad_prior_beta;grad_prior_theta_gammaj;grad_prior_theta_sigma2];
        grad_h = grad_jacobian+grad_prior+grad_llh;
        grad_g = [eye(d);epsilon1*eye(d);diag(epsilon2)];
        grad_g_lik_store(s,:)= (grad_g*grad_h)';      
    end    
    nabla_C = compute_nabla_C(b,c);
    nabla_D = (mean(grad_g_lik_store))';
    grad_lb = nabla_D-nabla_C;
    gradient_lambda = inverse_fisher_times_grad(b,c,grad_lb);

    grad_norm_current = norm(gradient_lambda);
    norm_gradient_seq1(iter) = grad_norm_current;
    norm_gradient_threshold = 50;
    if norm(gradient_lambda)>norm_gradient_threshold
        gradient_lambda = (norm_gradient_threshold/norm(gradient_lambda))*gradient_lambda;
    end
    norm_gradient = norm_gradient+norm(gradient_lambda);
         
    gradient_bar_old = gradient_bar;     
    gradient_bar = grad_weight*gradient_bar+(1-grad_weight)*gradient_lambda;
    
    if iter>tau
        stepsize = lrate*tau/iter;
    else
        stepsize = lrate;
    end
    
    % Extract mu,b,c to recalculate loss
    lambda = lambda+stepsize*gradient_bar;    
    mu = lambda(1:d,1);
    b = lambda(d+1:2*d,1);
    c = lambda(2*d+1:end);
    
    % Update prior hyper-parameter after each 4 (tunable) iterations
    if mod(iter,4) == 0
        mu_w_tilde = mu(w_tilde_index); 
        b_w_tilde = b(w_tilde_index);
        c_w_tilde = c(w_tilde_index);
        mu_beta_tilde = mu(beta_tilde_index); 
        b_beta_tilde = b(beta_tilde_index);
        c_beta_tilde = c(beta_tilde_index);
        mean_beta_tilde = mu_beta_tilde'*mu_beta_tilde+b_beta_tilde'*b_beta_tilde+sum(c_beta_tilde.^2);
        gamma_beta = n_units(end)/mean_beta_tilde;
        mean_w_tilde = mu_w_tilde'*mu_w_tilde+b_w_tilde'*b_w_tilde+sum(c_w_tilde.^2);
        gamma_w = length(w_tilde_index)/mean_w_tilde;
    end
    
    W1 = reshape(mu(1:index_track(1)),n_units(1),p+1);
    W_seq{1} = W1;
    W1_tilde = W1(:,2:end); % weights without biases                
    grad_prior_w = [zeros(n_units(1),1);-gamma_w*W1_tilde(:)];        
    for j = 2:L
        index = index_track(j-1)+1:index_track(j);
        Wj = reshape(mu(index),n_units(j),n_units(j-1)+1);
        W_seq{j} = Wj; 
    end
    beta = mu(d_w+1:d_w+d_beta);
    
    % Calculate loss at current iteration
    if(strcmp(distr,'normal'))
        sigma2 = exp(mu(end)); % for error variance
        theta_gammaj = mu(d_w+d_beta+1:end-1);
        loss =  prediction_loss(W_seq,beta,theta_gammaj,y_val,X_val,y,X,distr,sigma2);
        Loss_DL(iter) = loss.pps;
        % Display current training results
        disp(['Iteration: ',num2str(iter),'   -  MSE: ',num2str(Loss_DL(iter))]);
    elseif (strcmp(distr,'binomial'))
        theta_gammaj = mu(d_w+d_beta+1:end);
        loss =  prediction_loss(W_seq,beta,theta_gammaj,y_val,X_val,y,X,distr);
        Loss_DL(iter) = loss.pps;
        % Display current training results
        disp(['Iteration: ',num2str(iter),'   -  PPS: ',num2str(Loss_DL(iter))]);
    else
    end
    
    if Loss_DL(iter)>=Loss_DL(iter-1)
        gradient_bar = gradient_bar_old;
    end
    
    if Loss_DL(iter)<=min(Loss_DL)
        lambda_best = lambda;
        idxPatience = 0;
    else
        idxPatience = idxPatience+1;
    end
    
    if (idxPatience>patience)||(iter>max_iter) 
        stop = true; 
    end 
end

%% Calculate
lambda = lambda_best;
mu = lambda(1:d,1);
b = lambda(d+1:2*d,1);
c = lambda(2*d+1:end);
W1 = reshape(mu(1:index_track(1)),n_units(1),p+1);
W_seq{1} = W1;
W1_tilde = W1(:,2:end); % weights without biases                
grad_prior_w = [zeros(n_units(1),1);-gamma_w*W1_tilde(:)];        
for j = 2:L
    index = index_track(j-1)+1:index_track(j);
    Wj = reshape(mu(index),n_units(j),n_units(j-1)+1);
    W_seq{j} = Wj; 
end
beta = mu(d_w+1:d_w+d_beta);

%% --------------------------Display Training Results----------------------
disp('---------- Training Completed! ----------')
disp(['Number of iteration:',num2str(iter)]);
if(strcmp(distr,'normal'))
    theta_gammaj = mu(d_w+d_beta+1:end-1);
    sigma2 = exp(mu(end));
    Loss_DL_best = prediction_loss(W_seq,beta,theta_gammaj,y_val,X_val,y,X,distr,sigma2);
    disp(['MSE best: ',num2str(Loss_DL_best.pps)]);
else
    theta_gammaj = mu(d_w+d_beta+1:d_w+d_beta+1+n_units(end));
    Loss_DL_best = prediction_loss(W_seq,beta,theta_gammaj,y_val,X_val,y,X,distr);    
    disp(['PPS best: ',num2str(Loss_DL_best.pps)]);
end

%% Store results
est.out.weights = W_seq; 
est.out.beta = beta;
est.out.theta_gammaj = theta_gammaj;
est.out.iteration = iter;
est.out.vbMU = mu;            % Mean of variational distribution of weights
est.out.b = b;
est.out.c = c;
est.out.nparams = d;
est.out.indexTrack = index_track;
est.out.best_accuracy = Loss_DL_best;
est.out.loss = Loss_DL;
if(strcmp(distr,'normal'))
    est.out.sigma2 = sigma2;
end
end

