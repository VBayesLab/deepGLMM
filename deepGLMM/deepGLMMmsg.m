function msg_out = deepGLMMmsg(identifier)
%DEEPGLMMMSG Define custom error/warning messages for exceptions
%   DEEPGLMMMSG = (IDENTIFIER) extract message for input indentifier
%   
%
%   Copyright 2018 Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) and Nghia
%   Nguyen (nghia.nguyen@sydney.edu.au)
%
%   https://github.com/VBayesLab/deepGLMM
%
%   Version: 1.0
%   LAST UPDATE: April, 2018

switch identifier
    case 'deepGLMM:TooFewInputs'
        msg_out = 'At least two arguments are specified';
    case 'deepGLMM:InputSizeMismatchX'
        msg_out = 'X and Y must have the same number of observations';
    case 'deepGLMM:InputSizeMismatchY'
        msg_out = 'Y must be a single column vector';
    case 'deepGLMM:ArgumentMustBePair'
        msg_out = 'Optinal arguments must be pairs';
    case 'deepGLMM:ResponseMustBeBinary'
        msg_out = 'Two level categorical variable required';
    case 'deepGLMM:DistributionMustBeBinomial'
        msg_out = 'Binomial distribution option required';
    case 'deepGLMM:MustSpecifyActivationFunction'
        msg_out = 'Activation function type requied';
end
end

