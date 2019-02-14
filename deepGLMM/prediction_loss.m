function loss = prediction_loss(W_seq,beta,theta_gammaj,ytest,Xtest,y,X,distr,varargin)

y_test_all = vertcat(ytest{:});
n = length(ytest);
ntest_all = length(y_test_all);
if strcmp(distr,'binomial')            % Binomial response
    prediction_error = 0;
    classification_rate = 0;
    for i = 1:n
        yi = y{i};
        Xi = X{i};    
        mu_alpha_i = newton_raphson(W_seq,beta,theta_gammaj,yi,Xi,distr);

        yi_test = ytest{i};
        Xi_test = Xtest{i};
        Ti_test = size(yi_test,1);
        for t = 1:Ti_test
            x_it = Xi_test(t,:)';
            y_it = yi_test(t);
            node_store = neural_net_output(x_it',W_seq,beta,mu_alpha_i);
            eta_it = node_store{end};
            % Calculate classification rate
            prediction_y_it = 0;
            if eta_it>=0 
                prediction_y_it = 1; 
            end   
            classification_rate = classification_rate+abs(y_it-prediction_y_it);
            % Calculate pps
            p_it = 1/(1+exp(-eta_it));
            prediction_error = prediction_error + CRPS(y_it,p_it);
        end
    end
    loss.pps = prediction_error/ntest_all;
    loss.classification_rate = classification_rate/ntest_all;
elseif strcmp(distr,'normal')          % Normal response
    f = 0;
    MSE = 0;
    sigma2 = varargin{1};
    for i = 1:n
        yi = y{i};
        Xi = X{i};    
        mu_alpha_i = newton_raphson(W_seq,beta,theta_gammaj,yi,Xi,distr,sigma2);

        yi_test = ytest{i};
        Xi_test = Xtest{i};
        Ti_test = size(yi_test,1);
        for t = 1:Ti_test
            x_it = Xi_test(t,:)';
            y_it = yi_test(t);
            node_store = neural_net_output(x_it',W_seq,beta,mu_alpha_i);
            eta_it = node_store{end};
            % Calculate pps
            f = f + 1/2*log(sigma2)+1/2/sigma2*(y_it-eta_it)^2;
            % Calculate mean square error
            MSE = MSE+(y_it-eta_it)^2;
        end
    end
    loss.pps = f/ntest_all;
    loss.mse = MSE/ntest_all;
else
    disp('Distribution must be either Normal or Binomial')
end

end



    
