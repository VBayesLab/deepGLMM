function output = activation(z,text)
% if strcmp(type,'tanh')    
%     f = exp(-2*z); f = (1-f)./(1+f);
% end
% if strcmp(type,'ReLU')
%     f = max(0,z);
% end

% Calculate activation output of hidden units
% z: pre-activation of current hidden unit -> can be a scalar or array
% vector (all units of a single hidden layer)
% text: specified activation function
% text = {Sigmoid, Tanh, ReLU, LeakyReLU, Maxout}

    global alpha   % Hyper parameter need to be tuned

    switch text
        case 'Linear'
            output = z;
        case 'Sigmoid'
            output = 1.0 ./ (1.0 + exp(-z));
        case 'Tanh'
            output = tanh(z);
        case 'ReLU'
            output = max(0,z);
        case 'LeakyReLU'
            output = max(0,z)+ alpha*min(0,z);
        case 'Maxout'
            
    end
end
