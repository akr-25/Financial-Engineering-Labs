dx = 0.75;
dt = dx^2 / 80;
[x_min, x_max, T] = deal(0, 30, 1);
[r, sigma, delta, K] = deal(0.06, 0.3, 0, 10);
N = ceil((x_max - x_min) / dx);
M = ceil(T / dt);
x = linspace(x_min, x_max, N + 1);
t = linspace(0, T, M + 1);
lambda = dt / (dx ^ 2);
alpha = b(x(1), t) * dt / dx ^ 2 - 0.5 * c(x(1), t) * dt / dx;
gamma = b(x(end - 1), t) * dt / dx ^ 2 + 0.5 * c(x(end - 1), t) * dt / dx;

% CrankNicolson
U = CrankNicolson(N, M, x, flip(t), dx, -dt, @f1, @f2, @g, @b, @c, @d, alpha, gamma);
U_plus_r = CrankNicolson(N, M, x, flip(t), dx, -dt, @f1, @f2_p, @g, @b, @c_p, @d_p, alpha, gamma);
U_minus_r = CrankNicolson(N, M, x, flip(t), dx, -dt, @f1, @f2_m, @g, @b, @c_m, @d_m, alpha, gamma);
U_plus_v = CrankNicolson(N, M, x, flip(t), dx, -dt, @f1, @f2, @g, @b_p, @c, @d, alpha, gamma);
U_minus_v = CrankNicolson(N, M, x, flip(t), dx, -dt, @f1, @f2, @g, @b_m, @c, @d, alpha, gamma);
[~, m] = size(U);

greek_delta = get_delta(U, dx); 
gamma = get_gamma(U, dx); 
theta = get_theta(U, dt); 
vega = get_vega(U_plus_v, U_minus_v, 0.01);
rho = get_rho(U_plus_r, U_minus_r, 0.001);

figure;

subplot(2, 3, 1);
plot(x(2:m-1), greek_delta(1, :), 'LineWidth', 2.3);
title('delta');

subplot(2, 3, 2);
plot(x(2:m-1), gamma(1, :), 'LineWidth', 2.3);
title('gamma');

subplot(2, 3, 3);
plot(x(2:m-1), theta(1, :), 'LineWidth', 2.3);
title('theta');

subplot(2, 3, 4);
plot(x(2:m-1), vega(1, :), 'LineWidth', 2.3);
title('vega');

subplot(2, 3, 5);
plot(x(2:m-1), rho(1, :), 'LineWidth', 2.3);
title('rho');

suptitle('Greek Plots using CrankNicolson');


function var = f1(~, t) 
    var = zeros(size(t));
end

function var = f2(x, t)
    K = 10;
    r = 0.06;
    T = 1;
    var = x - K * exp(-r * (T - t));  
end


function var = f2_p(x, t)
    K = 10;
    r = 0.06 + 0.001;
    T = 1;
    var = x - K * exp(-r * (T - t));  
end


function var = f2_m(x, t)
    K = 10;
    r = 0.06 - 0.001;
    T = 1;
    var = x - K * exp(-r * (T - t));  
end

function var = g(x)
    K = 10;
    tmp = x - K;
    var = tmp .* (tmp >= 0);
end

function var = a(~, ~)
    var = 1;
end

function var = b(x, ~)
    sigma = 0.3;
    var = 0.5 * (sigma ^ 2) * (x .^ 2);
end

function var = b_p(x, ~)
    sigma = 0.3 + 0.01;
    var = 0.5 * (sigma ^ 2) * (x .^ 2);
end

function var = b_m(x, ~)
    sigma = 0.3 - 0.01;
    var = 0.5 * (sigma ^ 2) * (x .^ 2);
end

function var = c(x, ~)
    r = 0.06;
    delta = 0;
    var = (r - delta) .* x;
end

function var = c_p(x, ~)
    r = 0.06 + 0.001;
    delta = 0;
    var = (r - delta) .* x;
end

function var = c_m(x, ~)
    r = 0.06 - 0.001;
    delta = 0;
    var = (r - delta) .* x;
end

function var = d(~, ~)
    r = 0.06;
    var = -r;
end

function var = d_p(~, ~)
    r = 0.06 + 0.001;
    var = -r;
end

function var = d_m(~, ~)
    r = 0.06 - 0.001;
    var = -r;
end