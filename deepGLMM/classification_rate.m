function [f,prediction] = classification_rate(W_seq,beta,theta_gammaj,ytest,Xtest,y,X)

n = length(ytest);
prediction_error = 0;
n_total = 0;
prediction = [];
for i = 1:n
    yi = y{i};
    Xi = X{i};    
    mu_alpha_i = newton_raphson(W_seq,beta,theta_gammaj,yi,Xi);
   
    yi_test = ytest{i};
    Xi_test = Xtest{i};
    Ti_test = size(yi_test,1);
    for t = 1:Ti_test
        x_it = Xi_test(t,:)';
        y_it = yi_test(t);
        node_store = neural_net_output(x_it',W_seq,beta,mu_alpha_i);
        eta_it = node_store{end};
        prediction_y_it = 0;
        if eta_it>=0 
            prediction_y_it = 1; 
        end       
        prediction = [prediction;prediction_y_it];
        prediction_error = prediction_error+abs(y_it-prediction_y_it);
    end
    n_total = n_total+Ti_test;
end
f = prediction_error/n_total;
end



    
