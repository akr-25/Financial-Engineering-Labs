id = 0;  % For unique image naming

k = 0.045; % Time 
h = 0.2; % space
sigma = 0.3;
T = 1;
r = 0.06;
lambda = k/h*h;
disp(lambda);
tmax = T*sigma*sigma/2;
xmax = 5;
xmin = -5;
n = nearest(tmax/k);
disp(n);
m = (xmax - xmin)/h; 
delta = 0;
q_delta = 2*(r - delta)/sigma^2;

% Spatial and temporal domains
temporalDomain = (0:n)*k; 
spatialDomain = (0:m)*h;

% Solve using various numerical schemes
solutionFTCS = FTCS(h, k, m, n, @fun, @f, @g1, @g2, xmin);
solutionBTCS = BTCS(h, k, m, n, @fun, @f, @g1, @g2, xmin);
solutionCN = CN(h, k, m, n, @fun, @f, @g1, @g2, xmin); 

% Plot and save for FTCS
generatePlot(spatialDomain, solutionFTCS(end, :),'FTCS', h, k, id);
generateSurface(spatialDomain, temporalDomain, solutionFTCS,'FTCS', h, k, id);
id = id + 1;

% Plot and save for BTCS
generatePlot(spatialDomain, solutionBTCS(end, :),'BTCS', h, k, id);
generateSurface(spatialDomain, temporalDomain, solutionBTCS,'BTCS', h, k, id);
id = id + 1;

% Plot and save for Crank-Nicolson
generatePlot(spatialDomain, solutionCN(end, :), 'Crank-Nicolson', h, k, id);
generateSurface(spatialDomain, temporalDomain, solutionCN,'Crank-Nicolson', h, k, id);
id = id + 1;


% PDE's non-homogeneous term
function val = fun(~, ~)
    val = 0;
end

% Initial condition
function val = f(x)
    delta = 0;
    sigma = 0.3;
    r = 0.06;
    q_delta = 2*(r - delta)/sigma^2;
    val = max(exp(x*(q_delta + 1)/2) - exp(x*(q_delta - 1)/2), 0);
end

% Left boundary condition
function val = g1(~, ~)
    val = 0;
end

% Right boundary condition
function val = g2(x, t)
    delta = 0;
    sigma = 0.3;
    r = 0.06;
    q_delta = 2*(r - delta)/sigma^2;
    val = exp((q_delta + 1)*x/2 + 0.25*t*(q_delta + 1)^2);
end
