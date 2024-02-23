function U = FTCS(N, M, x, t, dx, dt, f1, f2, g, b, c, d, alpha, gamma)
    % This function solves the given general parabolic PDE using FTCS scheme.
    % OUTPUT: U (representing the solution matrix)
    

    % Create solution matrix with given ICs
    U = zeros(M + 1, N + 1);
    U(1, :) = g(x);
    U(:, 1) = f1(x(1), t);
    U(:, end) = f2(x(end), t);
    
    B = zeros(N - 1, 1); 
    % Prepare tri-diagonal matrix
    L = tridiagonal_matrix(N - 1, x, t, b, c, d, dx, dt, 1);
        
    % Compute the values of U(x, t) at grid points
    for j = 2 : M + 1
        B(1) = alpha * U(j - 1, 1); 
        B(end) = gamma * U(j - 1, end);
        U(j, 2 : end - 1) = L * (U(j - 1, 2 : end - 1)') + B;
    end

    U = flipud(U);
end