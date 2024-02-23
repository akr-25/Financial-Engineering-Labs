% Define step sizes for spatial and temporal discretization
hValues = [1e-2, 1e-3, 1e-4]; 
kValues = [5e-4, 1e-3, 1e-2]; 

id = 0;  % For unique image naming

% Loop through all combinations of h and k values
for i = 1:3
    k = kValues(i);
    h = hValues(i);
    n = 1/k;
    m = 1/h; 

    % Spatial and temporal domains
    temporalDomain = (0:n)*k; 
    spatialDomain = (0:m)*h;

    % Solve using various numerical schemes
    solutionFTCS = FTCS(h, k, m, n, @fun, @f, @g1, @g2);
    solutionBTCS = BTCS(h, k, m, n, @fun, @f, @g1, @g2);
    solutionCN = CN(h, k, m, n, @fun, @f, @g1, @g2);

    % Analytical solution for comparison
    analyticalSolution = exp(-(pi^2)) * sin(pi .* spatialDomain); 

    % Plot and save for FTCS
    generatePlot(spatialDomain, solutionFTCS(end, :), analyticalSolution, 'FTCS', h, k, id);
    generateSurface(spatialDomain, temporalDomain, solutionFTCS,'FTCS', h, k, id);
    id = id + 1;

    % Plot and save for BTCS
    generatePlot(spatialDomain, solutionBTCS(end, :), analyticalSolution, 'BTCS', h, k, id);
    generateSurface(spatialDomain, temporalDomain, solutionBTCS,'BTCS', h, k, id);
    id = id + 1;

    % Plot and save for Crank-Nicolson
    generatePlot(spatialDomain, solutionCN(end, :), analyticalSolution, 'Crank-Nicolson', h, k, id);
    generateSurface(spatialDomain, temporalDomain, solutionCN,'Crank-Nicolson', h, k, id);
    id = id + 1;

end


% PDE's non-homogeneous term
function val = fun(~, ~)
    val = 0;
end

% Initial condition
function val = f(x)
    val = sin(pi*x);
end

% Left boundary condition
function val = g1(~)
    val = 0;
end

% Right boundary condition
function val = g2(~)
    val = 0;
end
