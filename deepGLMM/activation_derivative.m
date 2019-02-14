function f = activation_derivative(z,type)
    % tanh
    if strcmp(type,'tanh')
        f = exp(-2*z); 
        f = (1-f)./(1+f); 
        f = 1-f.^2;
    end
    % rectified
    if strcmp(type,'ReLU')
        f = ones(size(z)); 
        f(z<=0) = 0;
    end
end