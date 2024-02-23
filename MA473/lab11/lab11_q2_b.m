

err = 500: 500: 10000; 
N = 500: 500: 10000;
for i = 1: length(N)
    del_t = 1/N(i); 
    timepts = 0 + del_t : del_t : 4 - del_t; 
    N_ITR = 200; 
    E = 1:N_ITR; 
    
    for j = 1 : N_ITR
        W = sqrt(del_t) * randn(length(timepts) , 1);
        X_tilde = second_order_milstein(10, 1, 0, del_t, timepts, W); 
        X_true =  f_Langevin(10, 1, 0, del_t, timepts, W); 
        E(j) = mean(abs(X_true - X_tilde));  
    end
    
    err(i) = mean(E); 
end

loglog(N, err, '-s'), hold on;
loglog(N, 1./N), hold off; 
legend('loglog N vs err', 'loglog N vs 1/N');


function X = f_Langevin(mu, sigma, X_init, del_t, timepts, W)
    X = zeros(size(timepts));
    n = length(timepts);
    
    for i = 1 : n
        if i == 1
            X(i) = exp(-mu*del_t)*(X_init + sigma*W(i));
        else
            X(i) = exp(-mu*del_t)*(X(i-1) + sigma*W(i));
        end
        
    end
end



function [X_tilde] = second_order_milstein(mu, sigma, X_init, del_t, timepts, W) 
    n = length(timepts); 
    X_tilde = timepts; 
    
    for i = 1:n
        if i == 1
            X_tilde(i) = X_init - mu*X_init*del_t + sigma*W(i) + ...
                - 0.5*mu*sigma*W(i)*del_t + 0.5*mu*mu*X_init*del_t*del_t;
        else
            X_tilde(i) = X_tilde(i-1) - mu*X_tilde(i-1)*del_t + sigma*W(i) + ...
                - 0.5*mu*sigma*W(i)*del_t + 0.5*mu*mu*X_tilde(i-1)*del_t*del_t;
        end
    end
end



