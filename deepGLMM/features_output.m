function Z_out = features_output(X,W_seq)
% compute the features (last hidden layer) of a 2-layer neural net 
% a1 = W1*X;
% Z1 = [1;max(0,a1)];
% a2 = W2*Z1;
% Z2 = [1;max(0,a2)];

[n_train, ~] = size(X);
n_layer = length(W_seq);
values = cell(1,n_layer+1);
% Save input values into first cell.
values{1} = X'; % each column of values{1} is a data row of X
% Apply neural network to input layer by layer.
for i = 1:n_layer
    z = W_seq{i} * values{i};%   [n_train,d(i)] x [d(i),d(i+1)]
    % Use identity activation function for output layer
    values{i+1} = activation(z,'ReLU');
    % Next, add biased to layer (i+1)
    values{i+1} = [ones(1,n_train);values{i+1}];
end
Z_out = values{end}';
end
