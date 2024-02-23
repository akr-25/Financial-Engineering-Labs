
% question 1 a 
err = 500: 500: 10000; 
N = 500: 500: 10000;
for i = 1: length(N)
    del_t = 1/N(i); 
    timepts = 0 + del_t : del_t : 1 - del_t; 
    N_ITR = 2000; 
    E = 1:N_ITR; 
    
    for j = 1 : N_ITR
        W = sqrt(del_t) * randn(length(timepts) , 1);
        X_tilde = BS_Milstein_second_order(0.75, 0.3, 307, del_t, timepts, W); 
        X_true =  f_BS(0.75, 0.3, 307, del_t, timepts, W); 
        E(j) = mean(abs(X_true - X_tilde));  
    end
    
    err(i) = mean(E); 
end

loglog(N, err, '-s'), hold on;
loglog(N, 1./N), hold off; 
title('milstein second order for black-scholes'); 
legend('loglog N vs err','loglog N vs 1/N');



function X = f_BS(mu, sigma, X_init, del_t, timepts, W)
    X = zeros(size(timepts));
    n = length(timepts);
    
    for i = 1 : n
        if i == 1
            X(i) = X_init * exp((mu - 0.5 * sigma * sigma) * del_t + sigma * W(i));
        else
            X(i) = X(i - 1) * exp((mu - 0.5 * sigma * sigma) * del_t + sigma * W(i));
        end
        
    end
end




function [X_hat] = BS_Milstein_second_order(mu, sigma, X_init, del_t, timepts, W) 
    n = length(timepts); 
    X_hat = timepts; 
    
    for i = 1:n
        if i == 1
            X_hat(i) = X_init + mu*X_init*del_t + sigma*X_init*W(i) + ...
                        0.5*(sigma^2)*X_init*((W(i)^2) - del_t) + ... 
                        0.5*mu*mu*X_init*del_t*del_t + ... 
                        mu*sigma*X_init*W(i)*del_t;
        else
            X_hat(i) = X_hat(i-1) + mu*X_hat(i-1)*del_t + sigma*X_hat(i-1)*W(i) + ...
                        0.5*(sigma^2)*X_hat(i-1)*((W(i)^2) - del_t) + ...
                        0.5*mu*mu*X_hat(i-1)*del_t*del_t + ... 
                        mu*sigma*X_hat(i-1)*W(i)*del_t;
        end
    end
end

