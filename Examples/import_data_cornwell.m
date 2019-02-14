clear
clc

% Example of preparing data for deepGLMMfit
% We store each individual data in a single cell 

% Read raw file
data_all = xlsread('cornwell&rupert.csv');    
data_temp = data_all(:,1:16);

% Changing the order of columns (Normally we put the last column as response)
data = [data_temp(:,16),data_temp(:,1:11),data_temp(:,12)];

N_sub = max(data(:,1));           % Number of subjects
n_train = 4;                      % For each subject, use first 4 observation for training
n_val = 1;                        % Next observation for validation
n_test = 2;                       % Last 2 observation for testing

% Pre-allocation
X = cell(1,N_sub);
X_validation = cell(1,N_sub);
X_test = cell(1,N_sub);
y = cell(1,N_sub);
y_validation = cell(1,N_sub);
y_test = cell(1,N_sub);

for i = 1:N_sub
    index = data(:,1)==i;
    panel_i = data(index,:);        
    T = size(panel_i,1);
    if T>=7
        y{i} = panel_i(1:n_train,end);
        y_validation{i} = panel_i(n_train+1:n_train+n_val,end);
        y_test{i} = panel_i(end-n_test+1:end,end);
        X{i} = [ones(n_train,1),panel_i(1:n_train,2:end-1)];
        X_validation{i} = [ones(n_val,1),panel_i(n_train+1:n_train+n_val,2:end-1)];
        X_test{i} = [ones(n_test,1),panel_i(n_train+n_val+1:end,2:end-1)];
        i = i+1;
    end
end

% Normalization
XX = vertcat(X{:});
mean_X = mean(XX); 
mean_X(1) = 0; 
mean_X(12) = 0; 
mean_X(4:10) = 0;
std_X = std(XX); 
std_X(1) = 1; 
std_X(12) = 1; 
std_X(4:10) = 1;
for i = 1:N_sub
    T_train = size(X{i},1);
    X{i} = X{i}-(ones(T_train,1)*mean_X);
    X{i} = X{i}./(ones(T_train,1)*std_X);
    
    T_validation = size(X_validation{i},1);
    X_validation{i} = X_validation{i}-(ones(T_validation,1)*mean_X);
    X_validation{i} = X_validation{i}./(ones(T_validation,1)*std_X);

    T_test = size(X_test{i},1);
    X_test{i} = X_test{i}-(ones(T_test,1)*mean_X);
    X_test{i} = X_test{i}./(ones(T_test,1)*std_X);
end

% Store processed data
save data_cornwell X y X_validation y_validation X_test y_test