% Take input variables for the PDE
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


% 1) FTCS
U = FTCS(N, M, x, flip(t), dx, -dt, @f1, @f2, @g, @b, @c, @d, alpha, gamma);
generatePlot(x, U(end, :), 'FTCS', dx, dt);
generateSurface(x, t, U, 'FTCS', dx, dt);

% 2) BTCS
U = BTCS(N, M, x, flip(t), dx, -dt, @f1, @f2, @g, @b, @c, @d, alpha, gamma);
generatePlot(x, U(end, :), 'BTCS', dx, dt);
generateSurface(x, t, U, 'BTCS', dx, dt);

% 3) CN
U = CrankNicolson(N, M, x, flip(t), dx, -dt, @f1, @f2, @g, @b, @c, @d, alpha, gamma);
generatePlot(x, U(end, :), 'CrankNicolson', dx, dt);
generateSurface(x, t, U, 'CrankNicolson', dx, dt);


function var = f1(~, t) 
    var = zeros(size(t));
end

function var = f2(x, t)
    K = 10;
    r = 0.06;
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

function var = c(x, ~)
    r = 0.06;
    delta = 0;
    var = (r - delta) .* x;
end

function var = d(~, ~)
    r = 0.06;
    var = -r;
end