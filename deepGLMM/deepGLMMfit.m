function est = deepGLMMfit(X,y,X_val,y_val,varargin)
% DEEPGLMMFIT Traing a deepGLMM model. deepGLMMfit is a flexible version of Generalized 
% Liner Mixed Model where Deep Feedforward Network is used to automatically choose 
% transformations for the raw covariates. Inverse gamma prior is used for 
% sigma2
% 
%   MDL = deepGLMMFIT(X,Y) fits a deepGLMM model using the design matrix X and 
%   response vector Y, and returns an output structure mdl to make prediction 
%   on a test data. By default, if 'distribution' option is not specified, 
%   deepGLMMfit will treat response variable y as normal distributed variable.
%
%   MDL = deepGLMMFIT(X,Y,NAME,VALUE) fit a deepGLMM model with additional options
%   specified by one or more of the following NAME/VALUE pairs:
%
%      'Distribution'     Name of the distribution of the response, chosen
%                         from the following:
%                 'normal'             Normal distribution (default)
%                 'binomial'           Binomial distribution
%      'Network'          Deep FeedforwardNeuron Network structure for deepGLMM. 
%                         In the current version, deepGLMM supports only 1 node 
%                         for the output layer, users just need to provide a 
%                         structure for hidden layers in an array where each 
%                         element in the array is the 
%                         number of nodes in the corresponding hidden layer.
%      'Lrate'            Vector of integer or logical indices specifying
%                         the variables in TBL or the columns in X that
%                         should be treated as categorical. Default is to
%                         treat TBL variables as categorical if they are
%                         categorical, logical, or char arrays, or cell
%                         arrays of strings.
%      'Momentum'         Momentum weight for stochastic gradient ascend. 
%                         The momentum determines the contribution of the 
%                         gradient step from the previous iteration to the 
%                         current iteration of training. It must be a value 
%                         between 0 and 1, where 0 will give no contribution 
%                         from the previous step, and 1 will give a maximal 
%                         contribution from the previous step. Must be between 
%                         0 and 1. 
%      'MaxIter'          The maximum number of iterations that will be used for 
%                         training. Must be a positive integer.
%      'Patience'         Number of consecutive times that the validation loss 
%                         is allowed to be larger than or equal to the previously 
%                         smallest loss before network training is stopped, 
%                         used as an early stopping criterion. Must be a positive 
%                         integer.
%      'LrateFactor'      Down-scaling factor that is applied to the learning 
%                         rate every time a certain number of iterations has 
%                         passed. Must be a positive integer
%      'S'                The number of samples needed for Monte Carlo 
%                         approximation of gradient of lower bound. Must 
%                         be an positive integer
%      'Seed'             Seeds the random number generator using the nonnegative 
%                         integer. Must be a nonnegative integer.
%
%   Example:
%      Fit a deepGLMM model for Direcmarketing data set. All of the
%      exampled data are located inside /Data folder of installed package. 
%      In order to use the sample dataset, user must add this Data folder
%      to Matlab path or explicitly direct to Data folder in 'load'
%      function
%
%      load('DirectMarketing.mat')
%      mdl = deepGLMMfit(X,y,...                   % Training data
%                        'Network',[5,5],...       % Use 2 hidden layers
%                        'Lrate',0.01,...          % Specify learning rate
%                        'MaxIter',10000,...       % Maximum number of epoch
%                        'Patience',50,...         % Higher patience values could lead to overfitting
%                        'Seed',100);              % Set random seed to 100
%
%   For more examples, check EXAMPLES folder
%
%   See also deepGLMMPREDICT
%
%   Copyright 2018:
%       Nghia Nguyen (nghia.nguyen@sydney.edu.au)
%       Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)
%      
%   https://github.com/VBayesLab/deepGLMM
%
%   Version: 1.0
%   LAST UPDATE: Feb, 2019


% Initialize output structure with default setting
est = deepGLMMout();

% Check errors input arguments
if nargin < 4
    error(deepGLMMmsg('deepGLMM:TooFewInputs'));
end
% if size(y,1) ~= size(X,1)
%     error(deepGLMMmsg('deepGLMM:InputSizeMismatchX'));
% end
% if size(y,2) ~= 1
%     error(deepGLMMmsg('deepGLMM:InputSizeMismatchY'));
% end
% if any(isnan(y)) || any(any(isnan(X))) % Check if data include NaN
%     error('NaN values not allowed in input data');
% end

if ~isempty(varargin)
    if mod(length(varargin),2)==1 % odd, model followed by pairs
        error(deepGLMMmsg('deepGLMM:ArgumentMustBePair'));
    end
end

%% Parse additional options
paramNames = {'S'               'Lrate'         'Initialize'      'Ncore'       ...
              'MaxIter'         'LRateFactor'   'Patience'        'Network'     ...
              'Distribution'    'Seed'          'Cvar'            'Bvar'        ...
              'Momentum'        'Verbose'       'BinaryCutOff'    'QuasiMC'     ...
              'MuTau'   };
          
paramDflts = {est.S             est.lrate       est.initialize    est.ncore     ...
              est.maxiter       est.tau         est.patience      est.network   ...
              est.dist          est.seed        est.c             est.bvar      ...
              est.momentum      est.verbose     est.cutoff        est.quasiMC   ...
              est.muTau};
[S,lrate,initialize,ncore,maxIter,tau,patience,...
 network,dist,seed,cvar,bvar,momentum,verbose,cutoff,...
 quasiMC,muTau] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});

% % Check errors for additional options
% % If distribution is 'binomial' but responses are not binary/logical value
% if (~isBinomial(y) && strcmp(dist,'binomial'))
%     error(deepGLMMmsg('deepGLMM:ResponseMustBeBinary'));
% end
% % If response is binary array but distribution option is not 'binomial'
% if (isBinomial(y) && ~strcmp(dist,'binomial'))
%     error(deepGLMMmsg('deepGLMM:DistributionMustBeBinomial'));
% end


%% Store training settings
est.S = S;
est.lrate = lrate;
est.initialize = initialize;
est.maxiter = maxIter;
est.tau = tau;
est.patience = patience;
est.network = floor(network);
est.dist = dist;
est.seed = seed;
est.cvar = cvar;
est.cutoff = cutoff;
est.bvar = bvar;
est.momentum = momentum;
est.verbose = verbose;
est.quasiMC = quasiMC;
est.muTau = muTau;
est.data.X = X;
est.data.y = y;
est.data.X_val = X_val;
est.data.y_val = y_val;

% Run training using Matlab scripts
tic
est = deepGLMMTrain(X,y,X_val,y_val,est);
CPU = toc;
disp(['Training time: ',num2str(CPU),'s']);
est.out.CPU = CPU;      % Save training time

end

