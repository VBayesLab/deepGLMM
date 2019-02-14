function values_store = neural_net_output(X,W_seq,beta,alpha_i)

[n_train, ~] = size(X);  % n_train: number of data row in training data
                                         % n_feature: number of features
weights = [W_seq,(beta+alpha_i)'];
n_layer = length(weights);   % number of layers in network  
                             %(not inluding input layer)

values = cell(1,n_layer+1);  % values(i) contain all the activation
                             % values of all units in layer i
                             % +1 -> first cell is for input layer

% Save input values into first cell.
values{1} = X'; % each column of values{1} is a data row of X
values_store = values;
% Add biases to input layer. First rows of input matrix is all 1
% values{1} = [ones(1,n_train);values{1}];

% Apply neural network to input layer by layer.
for i = 1:n_layer
    % Each values{i} is a matrix where each column stores all activated
    % units of layer i of corresponding input data row
    % Ex: Column 3th of values{i} store activated units of layer i
    % of 3th input data row

    % First, calculate activation values for all unit in next layer
    % values{2} -> layer 1
    z = weights{i} * values{i};%   [n_train,d(i)] x [d(i),d(i+1)]

    % Use identity activation function for output layer
    if i == n_layer
        values{i+1} = activation(z,'Linear');
        values_store{i+1} = values{i+1};
    else  % if current layer is a hidden layer, use activation function
          % specified by text argument
        values{i+1} = activation(z,'ReLU');
        values_store{i+1} = values{i+1};
        % Next, add biased to layer (i+1)
        values{i+1} = [ones(1,n_train);values{i+1}];
    end
end

end
