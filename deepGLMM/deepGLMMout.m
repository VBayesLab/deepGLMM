function out = deepGLMMout()
%DEEPGLMMOUT Generate default output structure for deepGLMM training results

%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%   
%   https://github.com/VBayesLab/deepGLMM
%
%   Version: 1.0
%   LAST UPDATE: Feb, 2019

    % Training method
    out.dist = 'normal';         % Default distribution of dependent variable. 
    out.initialize = 'adaptive'; % Default initialize method
    out.isIsotropic = false;     % Default structure of variational Covariance matrix
    out.ncore = 0;               % Default parallel computing option
    
    % Optional settings
    out.seed = NaN;              % No random seed by default
    out.nval = 0.2;              % Default proportion of training data for validation
    out.verbose = true;          % Default number of iteration to display training results
    out.cutoff = 0.5;            % Default Cutoff probability for sigmoid function
    out.stop = false;            % Execution Flag
    out.quasiMC = true;          % Using Quasi MC for random number generator
    out.muTau = NaN;
    
    % Model hyper-parameters
    out.network = [5,5];         % Default network structure
    out.lrate = 0.1;             % Default Learning rate
    out.S = 10;                  % Default Number of samples used to approximate gradient of likelihood
    out.maxiter = 300;           % Default Number of maximum iteration in train phase
    out.tau = 100;               % Default Scale factor of learning rate
    out.patience = 10;           % Default Number of consequence non-decreasing iterations (for early stopping checking)
    out.c = 0.01;                % Default initial value of isotropic factor c
    out.bvar = 0.01;             % Default initial variance of each element of b
    out.momentum = 0.6;          % Default momentum weight
    out.gamma_w = 50; 
    out.gamma_beta = 50;
    out.hp_beta_0 = 0.1;
 
    % Variable names
    out.name.ynames = NaN;       % y variables names
    out.name.xnames = NaN;       % X variable names
    
    % Data properties
    out.data.y = NaN;            % Dependent variable of training data
    out.data.X = NaN;            % Independent variables of training data
    out.data.ytest = NaN;        % Dependent variable of test data
    out.data.Xtest = NaN;        % Independent variables of tets data
    out.data.nTrain = NaN;       % Number of observation in training set
    out.data.nTest = NaN;        % Number of observation in test set
    out.data.Xval = NaN;
    out.data.yval = NaN;
    out.data.icept = true;       % Intercept option
    
    % Training results
    out.out.mse = NaN;
    out.out.accuracy = NaN;
    
end

