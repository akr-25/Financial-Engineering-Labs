schemes = { @(N, M, x, t, lambda, dx, dt, T, r, sigma, delta, f1, f2, g, a, b, c, d, m, q) FTCS(N, M, x, t, lambda, dx, dt, T, r, sigma, delta, f1, f2, g, a, b, c, d, m, q),
            @(N, M, x, t, lambda, dx, dt, T, r, sigma, delta, f1, f2, g, a, b, c, d, m, q) BTCS(N, M, x, t, lambda, dx, dt, T, r, sigma, delta, f1, f2, g, a, b, c, d, m, q),
            @(N, M, x, t, lambda, dx, dt, T, r, sigma, delta, f1, f2, g, a, b, c, d, m, q) CrankNicolson(N, M, x, t, lambda, dx, dt, T, r, sigma, delta, f1, f2, g, a, b, c, d, m, q)};

for i = 1 : length(schemes)
    [dx, dt, x_min, x_max, T, r, sigma, delta, f1, f2, g, a, b, c, d, q] = input_pde();
    
    [x, t, N, M, lambda] = create_grid(dx, dt, x_min, x_max, T);
    
    if i == 1
        methods = [string('')];
    else
        methods = [string('right-matrix division')];
    end
    
    for m = methods
        
        [V, V_untranformed, S, S_untransformed, time] = schemes{i}(N, M, x, t, lambda*dx, dx, dt, T, r, sigma, delta, f1, f2, g, a, b, c, d, m, q);

        surface_plot(S, time, V, i, m);
        plot_final_time_level(S,time, V, i); 
    end
end


function [dx, dt, x_min, x_max, T, r, sigma, delta, f1, f2, g, a, b, c, d, q] = input_pde()
    
    dx = 0.05;
    dt = dx / 2;
    q = 0.05; 
    
    [x_min, x_max, T] = deal(0, 1, 1);
    [r, delta] = deal(0.04,  0.1);
    
    sigma = @(x) x/4; 
    sigma_bar = @(x) sigma((q*x)/(1-x));
    f1 = @(x, t, r, T) exp(-delta * t);  
    f2 = @(x, t, r, T) zeros(size(t));
    
    helper = @(x) 1 - 2 * x; 
    g = @(x) helper(x) .* (helper(x) >= 0);
    
    a = @(x, t) 1;
    b = @(x, t, sigma) -0.5 * (sigma_bar(x) ^ 2) * (x .^ 2) * (1 - x .^ 2);
    c = @(x, t, r, delta) -(r - delta) .* x .* (1 - x);
    d = @(x, t, r, delta) r * (1 - x) + delta * x;
end