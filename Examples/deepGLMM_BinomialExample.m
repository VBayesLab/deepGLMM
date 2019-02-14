% Examples demonstate how to use deepGLMM function to fit data with continuos 
% dependent variable
%
% Copyright 2018 
%                Nghia Nguyen (nghia.nguyen@sydney.edu.au)
%                Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au) 
%
% https://github.com/VBayesLab/deepGLMM
%
% Version: 1.0
% LAST UPDATE: Feb, 2019

clear
clc

% load data
load('../Data/data_sim_easy.mat')

%% Fit deepGLMM model using default setting
% By default, if 'distribution' option is not specified then deepGLMMfit
% will assign the response variables as 'normal'
nn = [5,5];
mdl = deepGLMMfit(X,y,...  
                  X_val,y_val,...
                  'Distribution','binomial',...
                  'Network',nn,... 
                  'Lrate',0.1,...           
                  'Verbose',1,...             % Display training result each iteration
                  'MaxIter',100,...
                  'Patience',10,...          % Higher patience values could lead to overfitting
                  'S',10,...
                  'Seed',100);

%% Prediction on test data
% Make prediction (point estimation) on a test set
% Make prediction on a test set without true response
Pred1 = deepGLMMpredict(mdl,X_test); 

% Make prediction on a test set with true response
Pred2 = deepGLMMpredict(mdl,X_test,y_test);                           
disp(['PPS on test data                : ', num2str(Pred2.pps)])
disp(['Classification rate on test data: ', num2str(Pred2.classification_rate)])