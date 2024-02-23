quadrature_rules = {@(f, x_min, x_max, h, N, i, j, a, b) simpsons(f, x_min, x_max, h, N, i, j, a, b) };
                 
basis_functions = { @(x_min, x_max, h, N, i, x) linear_basis(x_min, x_max, h, N, i, x) };

basis_derivatives = { @(x_min, x_max, h, N, i, x, flag) derivative_linear_basis(x_min, x_max, h, N, i, x, flag) };


[T, K] = deal(1, 10);
[r, sigma, delta] = deal(0.06, 0.3, 0);

q = 2 * r / (sigma ^ 2); 
q_delta = 2 * (r - delta) / (sigma ^ 2);

[x_min, x_max, N, M] = deal(-4, 2, 100, 80);
[x, tau, h, k] = create_grid(x_min, x_max, N, M, T, sigma);
x_eff = x(2 : length(x)-1); 

for q_idx = 1 : length(quadrature_rules)
    if q_idx == 1
        fprintf('Solving European Call option using linear basis & Simpsons Rule ...\n\n');
    end
    
    [quadrature, basis, der_b] = deal(quadrature_rules{q_idx}, basis_functions{1}, basis_derivatives{1});
    
    % Construct stiffness matrix A and mass Matrix B
    [A, B] = deal(zeros(length(x_eff)));
    for i = 1 : length(x_eff)
        for j = 1 : length(x_eff)
            if i == j
                for idx = 0 : 1
                    func1 = @(x_min, x_max, h, N, i, j, val, fl) der_b(x_min, x_max, h, N, i, val, fl) * der_b(x_min, x_max, h, N, j, val, fl);
                    A(i, j) = A(i, j) + quadrature(func1, x_min, x_max, h, length(x), i, j, x(i+idx), x(i+idx+1));

                    func3 = @(x_min, x_max, h, N, i, j, val, fl) basis(x_min, x_max, h, N, i, val) * basis(x_min, x_max, h, N, j, val);
                    B(i, j) = B(i, j) + quadrature(func3, x_min, x_max, h, length(x), i, j, x(i+idx), x(i+idx+1));
                end
            elseif abs(i - j) == 1
                idx = min(i, j);
                func1 = @(x_min, x_max, h, N, i, j, val, fl) der_b(x_min, x_max, h, N, i, val, fl) * der_b(x_min, x_max, h, N, j, val, fl);
                A(i, j) = A(i, j) + quadrature(func1, x_min, x_max, h, length(x), i, j, x(idx+1), x(idx+2));

                func3 = @(x_min, x_max, h, N, i, j, val, fl) basis(x_min, x_max, h, N, i, val) * basis(x_min, x_max, h, N, j, val);
                B(i, j) = B(i, j) + quadrature(func3, x_min, x_max, h, length(x), i, j, x(idx+1), x(idx+2));
            end
        end
    end
    
    B_final = B + (0.5 * k) .* A;
    A_final = B - (0.5 * k) .* A;

    [w, b] = deal(zeros(length(x_eff), length(tau)));

    for i = 1 : length(tau)
        b(:, i) = ((x_eff - x_min)/(x_max - x_min)) * h * (0.25 * (q_delta + 1) ^ 2) * beta(x_max, tau(i), q_delta);
    end

    % Construct weights w
    w(: , 1) = IC(x_eff, q_delta) - phi_b(x_eff, 0, q_delta, x_min, x_max);
    for i = 2 : length(tau)
        w(:, i) = B_final \ (A_final * w(:, i - 1) - (k/2) * (b(:, i) + b(:, i-1)));
    end

    % Get option price at nodes
    base = zeros(N + 1, M + 1);
    for i = 1 : length(x)
        for j = 1 : length(tau)
            base(i, j) = phi_b(x(i), tau(j), q_delta, x_min, x_max);
        end
    end

    temp = zeros(1, M + 1);
    nodes = [temp; w; temp];
    nodes = nodes + base;

    V = nodes;
    temp1 = (-0.5) * (q - 1) * x;
    temp2 = (-0.25) * ((q + 1)^2) * tau;

    for i = 1 : length(x)
        for j = 1 : length(tau)
            V(i, j) = (K * exp(temp1(i) + temp2(j))) * nodes(i, j);
        end
    end

    % Plot graphs
    line_plot(x, tau, V, T, sigma,K);
    surface_plot(x, tau, V, T, sigma, K, q_idx, 'European Call option using');
    
end


% ***************  HELPER FUNCTIONS  ***************

function y = IC(x, q_delta)
    y = max(0, exp(0.5 * x * (q_delta + 1)) - exp(0.5 * x * (q_delta - 1)));
end

% Boundary condition at x_min
function y = alpha(x_min, tau, q_delta)
    y = zeros(size(tau));
end

% Boundary condition at x_max
function y = beta(x_max, tau, q_delta)
    y = exp(0.5 * (q_delta + 1) * x_max + 0.25 * tau * (q_delta + 1) ^ 2);
end

function val = phi_b(x, tau, q_delta, x_min, x_max)
    val = beta(x_max, tau, q_delta) - alpha(x_min, tau, q_delta);
    val = alpha(x_min, tau, q_delta) + val * (x - x_min) / (x_max - x_min);
end

function val = simpsons(f, x_min, x_max, h, N, i, j, a, b) 
    val = (h / 6) * (f(x_min, x_max, h, N, i, j, a, 1) + 4*f(x_min, x_max, h, N, i, j, (a + b)/2, 0) + f(x_min, x_max, h, N, i, j, b, 0));
end

function val = linear_basis(x_min, x_max, h, N, i, x)
    [x1, x2, x3] = deal(x_min + (i-1)*h, x_min + i*h, x_min + (i+1)*h);
    
    if x < x1 || x > x3
        val = 0;
    else
        if i == N || x < x2
            val = phi(1, x_min, x_max, h, N, i-1, x, 'function-val');
        else
            val = 1 - phi(1, x_min, x_max, h, N, i, x, 'function-val');
        end
    end
end

function val = derivative_linear_basis(x_min, x_max, h, N, i, x, flag)
	[x1, x2, x3] = deal(x_min + (i-1)*h, x_min + i*h, x_min + (i+1)*h);

    if flag == 0
		x = x - h/3;
	else
		x = x + h/3;
    end
    
    if x < x1 || x > x3
        val = 0;
    else
        if i == N || x < x2
            val = phi(1, x_min, x_max, h, N, i-1, x, 'derivative');
        else
            val = -phi(1, x_min, x_max, h, N, i, x, 'derivative');
        end
    end
end


function [U] = Crank_Nicolson(fun, f, g1, g2, T, K, r, sig, delta, q, qd, x_min, x_max, h, k, m, n, X, Tau, method)
	lamda = k / h^2;
	U = zeros(n+1, m+1);

	U(1:end, 1) = g1(x_min, Tau, qd);
	U(1:end, end) = g2(x_max, Tau, qd);
	U(1, 1:end) = f(X, qd);

	A = zeros(m-1, m-1);
	B = zeros(m-1, m-1);

	for i = 2:n+1
		b = zeros(m+1, 1);

		A(1:m+2:end) = 1 + lamda;
		A(2:m+2:end) = -lamda/2;
		A(m+2:m+2:end) = -lamda/2;

		A(1,1) = 1;
		A(1,2) = 0;
		A(m+1,m+1) = 1;
		A(m+1,m) = 0;

		b(2:m) = U(i-1,1:m-1)*lamda/2 + (1-lamda)*U(i-1,2:m) + U(i-1,3:m+1)*lamda/2;
		b(1) = U(i,1);
		b(end) = U(i,end);

		U(i,:) = (A\b)';
	end

	U = transform(U, X, Tau, q, qd, K);
end

function val = phi(degree, x_min, x_max, h, N, i, x, type)
	[x1, x2] = deal(x_min + i*h, x_min + (i + degree)*h); 
	
    if x >= x1 && x <= x2
        if strcmp(type, 'derivative')
            val = 1 / h;
        else
            val = (x - x1) / h;
        end
	else
		val = 0;
    end
end

function [y] = transform(U, X, Tau, q, qd, K)
	y = zeros(size(U));
	if length(Tau) == 1
		for j = 1:length(X)
			y(j) = U(j) * K * exp(-0.5* (qd-1)*X(j) - (0.25*(qd-1)^2 + q)*Tau);
		end
	else
		for i = 1:length(Tau)
			for j = 1:length(X)
				y(i, j) = U(i, j) * K * exp(-0.5* (qd-1)*X(j) - (0.25*(qd-1)^2 + q)*Tau(i));
			end
		end
	end
end

function surface_plot(x, tau, V, T, sigma, K, idx, plot_title)
    figure();
    S = K * exp(x);
    time = T - tau * 2 / sigma^2;
    
    surf(S', time', V');
    
    xlabel('S');
    ylabel('t');
    zlabel('V(S, t)');
    
    
    title('Piecewise-linear basis functions with the Simpsons rule and the Crank-Nicolson scheme');
end

function [x, tau, h, k] = create_grid(x_min, x_max, N, M, T, sigma)
    x = linspace(x_min, x_max, N + 2);
    tau = linspace(0, T*(sigma^2)/2, M + 1); 
    [h, k] = deal(x(2) - x(1), tau(2) - tau(1));
end

function line_plot(x, tau, V, T, sigma,K)
    S = K * exp(x);
    time = T - tau * 2 / sigma^2;
   
    figure();
   
    plot(S,V(:,81), 'r');
    xlabel('S');
    ylabel('V(S,T)');
    title('European Call option using piecewise-linear basis functions with Simpson’s rule');
    legend('Numerical');
end
