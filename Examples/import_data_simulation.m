% Create a simulation panel dataset to run with deepGLMMfit

% Data Description:
%    - 1000 subjects
%    - 20 data points per subject
%         - 14 for training
%         - 3 for evaluation
%         - 3 for testing

% y_it = 5 + 3(x_it1 + 2*x_it2)^2 + 5*x_it3*x_it4 + 2*x_it5 + alpha_i + epsilon_it
% alpha_i ~ N(0,tau^2) 
% epsilon_it ~ N(0,sigma^2)
clear 
clc

n_var = 5;                  % Number of predictors
n_subs = 50;                % Number of subjects
T = 20;                     % Number of data points within a subject 
tau = 0.1;
sigma = 1;
X_data = cell(1,n_subs);
Y_data = cell(1,n_subs);
num_one = 0;
for i=1:n_subs              % For each group
    b_i = normrnd(0,tau,1); % Random effect within each subject
    Y_it = zeros(T,1);      % To store outputs within each subject
    X_it = zeros(T,n_var) ; % To store input within each subject
    for t=1:T
        epsilon_it = normrnd(0,sigma,1);
        X_it(t,:) = unifrnd(0,2,5,1);
        Y_it(t) = 2 + 3*X_it(t,1)^2-2*X_it(t,2)*X_it(t,3)+3*X_it(t,4) ...
        - X_it(t,5) + b_i + epsilon_it;
    end
    X_data{i} = X_it;
    Y_data{i} = Y_it;
end

% Divide to train + validation + test
n_train = 14;
n_val = 3;
n_test = 3;

X = cell(1,n_subs);
X_validation = cell(1,n_subs);
X_test = cell(1,n_subs);
y = cell(1,n_subs);
y_validation = cell(1,n_subs);
y_test = cell(1,n_subs);

for i=1:n_subs
    X{i} = [ones(n_train,1) X_data{i}(1:n_train,:)];
    X_validation{i} = [ones(n_val,1) X_data{i}(n_train+1:n_train+n_val,:)];
    X_test{i} = [ones(n_test,1) X_data{i}(end-n_test+1:end,:)];
    
    y{i} = Y_data{i}(1:n_train);
    y_validation{i} = Y_data{i}(n_train+1:n_train+n_val);
    y_test{i} = Y_data{i}(end-n_test+1:end);
end

save data_sim_easy X X_validation X_test y y_validation y_test








