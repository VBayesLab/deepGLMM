function gradient = nn_backpropagation(X,Y_train,W_seq,beta,alpha_i,distr)
% compute the gradient in a L-layer neural net using backpropagation

n_train = size(X,1);
L = length(W_seq);
a_seq = cell(1,L);
Z_seq = cell(1,L);
aux = (beta+alpha_i);

a_seq{1} = W_seq{1}*X';
Z_seq{1} = [ones(1,n_train);activation(a_seq{1},'ReLU')];
for j=2:L
    a_seq{j} = W_seq{j}*Z_seq{j-1};
    Z_seq{j} = [ones(1,n_train);activation(a_seq{j},'ReLU')];
end

delta_seq = cell(1,L+1);
Y = aux'*Z_seq{L};

if strcmp(distr,'normal')
    delta_seq{L+1} = (Y_train - Y')';
else
    p_i = 1./(1+exp(-Y));
    delta_seq{L+1} = (Y_train - p_i')';
end


delta_seq{L} = (aux(2:end)*delta_seq{L+1}).*activation_derivative(a_seq{L},'ReLU');
for j=L-1:-1:1
    Wj_tilde = W_seq{j+1};
    Wj_tilde = Wj_tilde(:,2:end);
    delta_seq{j} = (activation_derivative(a_seq{j},'ReLU')).*(Wj_tilde'*delta_seq{j+1});
end
gradient_W1 = delta_seq{1}*X;
gradient = gradient_W1(:);
for j = 2:L
    gradient_Wj = delta_seq{j}*(Z_seq{j-1})';
    gradient = [gradient;gradient_Wj(:)];
end
gradient = [gradient;Z_seq{L}*delta_seq{L+1}'];


end
