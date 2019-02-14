function out = deepGLMMpredict(mdl,Xtest,varargin)
%DEEPGLMMPREDICT Make prediction from a trained deepGLMM model
%
%   OUT = DEEPGLMMPREDICT(MDL,XTEST) predict responses for new data XTEST using 
%   trained deepGLMM structure MDL (output from DEEPGLMMFIT) 
%
%   OUT = DEEPGLMMPREDICT(MDL,XTEST,YTEST) predicts responses with specified true response column of new 
%   observations. deepGLMMpredict will return prediction scores (PPS, MSE or Classification Rate) using true 
%   responses column vector ytest
%
%   Example:
%
%      load('data_cornwell.mat')
%      mdl = deepGLMfit(X,y,...                   % Training data
%                      'Network',[2,2],...        % Use 2 hidden layers
%                      'Lrate',0.1,...            % Specify learning rate
%                      'MaxIter',300,...          % Maximum number of epoch
%                      'Patience',5,...           % Higher patience values could lead to overfitting
%                      'Seed',100);               % Set random seed to 100
%    
%      Pred = deepGLMMpredict(mdl,X_test,y_test);
%      disp(['PPS on test data: ',num2str(Pred.pps)])
%      disp(['MSE on test data: ',num2str(Pred.mse)])
%   
%   For more examples, check EXAMPLES folder
%
%   See also DEEPGLMFIT
%
%   Copyright 2018:
%       Nghia Nguyen (nghia.nguyen@sydney.edu.au)
%       Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)
%      
%   https://github.com/VBayesLab/deepGLMM
%
%   Version: 1.0
%   LAST UPDATE: Feb, 2019

% Check errors input arguments
if nargin < 2
    error(deepGLMmsg('deepglmm:TooFewInputs'));
end

% Load deepGLMM params from struct
W_seq = mdl.out.weights;             % Weights from first to second-last hidden layer
beta = mdl.out.beta;                 % Weight from last hidden layer to output layer
theta_gammaj = mdl.out.theta_gammaj;
distr = mdl.dist;                    % Distribution of responses
ytest = [];                          % Response
mu_alpha = zeros(length(theta_gammaj),1);
n = length(Xtest);
% Check if true responses are specified
if(~isempty(varargin))
    ytest = varargin{1};
    y_test_all = vertcat(ytest{:});
    ntest_all = length(y_test_all);
end

if strcmp(distr,'binomial')            % Binomial response
    pps = 0;                           % Only available if true response is provide
    classification_rate = 0;           % Only available if true response is provide
    prediction = cell(1,n);            % Store prediction (binary vector)
    for i = 1:n   
        mu_alpha_i = mvnrnd(mu_alpha,diag(exp(theta_gammaj)));
        Xi_test = Xtest{i};
        Ti_test = size(Xtest,1);
        
        if(~isempty(ytest))
            yi_test = ytest{i};
        end
        prediction{i} = [];
        for t = 1:Ti_test
            x_it = Xi_test(t,:)';
            node_store = neural_net_output(x_it',W_seq,beta,mu_alpha_i');
            eta_it = node_store{end};
            p_it = 1/(1+exp(-eta_it));
            % Calculate classification rate
            prediction_y_it = 0;
            if eta_it>=0 
                prediction_y_it = 1; 
            end   
            prediction{i} = [prediction{i};prediction_y_it];
            if(~isempty(ytest))
                y_it = yi_test(t);
                % Calulcate classification rate
                classification_rate = classification_rate+abs(y_it-prediction_y_it);
                % Calculate pps
                pps = pps + CRPS(y_it,p_it);
            end
        end
    end
    if(~isempty(ytest))
        out.pps = pps/ntest_all;
       out.classification_rate = classification_rate/ntest_all;
    end
    out.prediction = prediction;
elseif strcmp(distr,'normal')          % Normal response
    f = 0;
    MSE = 0;
    sigma2 = mdl.out.sigma2;           % Variance of noise
    prediction = cell(1,n);
    for i = 1:n
        mu_alpha_i = mvnrnd(mu_alpha,diag(exp(theta_gammaj)));
        Xi_test = Xtest{i};
        Ti_test = size(Xi_test,1);
        if(~isempty(ytest))
            yi_test = ytest{i};
        end
        prediction{i} = [];
        for t = 1:Ti_test
            x_it = Xi_test(t,:)';
            node_store = neural_net_output(x_it',W_seq,beta,mu_alpha_i');
            eta_it = node_store{end};
            prediction{i} = [prediction{i};eta_it];
            if(~isempty(ytest))
               y_it = yi_test(t);
               % Calculate pps
               f = f + 1/2*log(sigma2)+1/2/sigma2*(y_it-eta_it)^2;
               % Calculate mean square error
               MSE = MSE+(y_it-eta_it)^2;
            end
        end
    end
    out.prediction = prediction;
    if(~isempty(ytest))
        out.pps = f/ntest_all;
        out.mse = MSE/ntest_all;
    end
else
    disp('Distribution must be either Normal or Binomial')
end
end

