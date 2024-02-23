% Parameters
T = 1;
K = 10;
r = 0.06;
sigma = 0.3;
delta = 0.05;
M = 100; % Space steps
N = 100; % Time steps
Smax = 2 * K; % maximum stock price
dt = T / N;
dS = Smax / M;
lambda = sigma^2 * dt / (dS^2);

% Initial Conditions
payoff = max(K - (0:dS:Smax), 0);

V_FTCS = zeros(M+1, N+1);
V_BTCS = zeros(M+1, N+1);
V_CN = zeros(M+1, N+1);

V_FTCS(:,1) = payoff;
V_BTCS(:,1) = payoff;
V_CN(:,1) = payoff;

% Tridiagonal matrix setup for BTCS and Crank-Nicolson
A_diag = (1 + lambda + r*dt + dS^2*sigma^2*dt/2);
A_off = 0.5*(-lambda + dS*delta - dS*r);
B_diag = (1 - lambda - r*dt - dS^2*sigma^2*dt/2);
B_off = 0.5*(lambda - dS*delta + dS*r);

A = diag(A_diag*ones(1,M+1)) + diag(A_off*ones(1,M),1) + diag(A_off*ones(1,M),-1);
B = diag(B_diag*ones(1,M+1)) + diag(B_off*ones(1,M),1) + diag(B_off*ones(1,M),-1);

V_CN = zeros(M+1, N+1);
tolerance = 1e-5;
max_iterations = 1000;

for j = 1:N
    % FTCS
    for i = 2:M
        V_FTCS(i,j+1) = V_FTCS(i,j) + lambda*(V_FTCS(i+1,j) - 2*V_FTCS(i,j) + V_FTCS(i-1,j)) + dt*r*S(i)*(V_FTCS(i+1,j) - V_FTCS(i,j))/2/dS - r*V_FTCS(i,j)*dt;
    end

    % Boundary Conditions for FTCS
    S_values = 0:dS:Smax;
    V_FTCS(1,j+1) = V_FTCS(1,j) + r*dt*(K*exp(-r*(T-j*dt)) - S_values(1));

    V_FTCS(M+1,j+1) = 0;  % Vanishing for large S

    % BTCS & CN
    rhs_BTCS = B * V_BTCS(:,j);

    V_BTCS(:,j+1) = A\rhs_BTCS;

    % Boundary Conditions for BTCS and CN
    S_values = 0:dS:Smax;
    V_BTCS(1,j+1) = V_BTCS(1,j) + r*dt*(K*exp(-r*(T-j*dt)) - S_values(1));

    V_BTCS(M+1,j+1) = 0;
        % Crank-Nicolson
end


% Plotting
time = 0:dt:T;
S = 0:dS:Smax;

figure;
surf(S, time, V_FTCS');
title('FTCS Solution');
xlabel('Stock Price');
ylabel('Time');
zlabel('Option Price');

figure;
surf(S, time, V_BTCS');
title('BTCS Solution');
xlabel('Stock Price');
ylabel('Time');
zlabel('Option Price');

figure;
surf(S, time, V_CN');
title('Crank-Nicolson Solution');
xlabel('Stock Price');
ylabel('Time');
zlabel('Option Price');
