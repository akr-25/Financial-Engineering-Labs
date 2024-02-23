% Generate diagonally dominant matrix A of size 5x5
n = 5;
A = rand(n, n); % create a random matrix

% Ensure A is diagonally dominant
for i = 1:n
    A(i,i) = A(i,i) + sum(abs(A(i,:))) - abs(A(i,i));
end

% Create random vector b of size n
b = rand(n, 1);

% Set tolerance and max iterations
tol = 1e-5;
max_iter = 1000;

% Call and print results from each method
[x_jacobi, iter_jacobi] = jacobi_method(A, b, tol, max_iter);
fprintf('Jacobi Method:\nSolution:\n');
disp(x_jacobi);
fprintf('Number of Iterations: %d\n', iter_jacobi);

[x_gs, iter_gs] = gauss_seidel_method(A, b, tol, max_iter);
fprintf('\nGauss-Seidel Method:\nSolution:\n');
disp(x_gs);
fprintf('Number of Iterations: %d\n', iter_gs);

omega = 1.25; % You can experiment with this value for optimal convergence
[x_sor, iter_sor] = sor_method(A, b, omega, tol, max_iter);
fprintf('\nSOR Method:\nSolution:\n');
disp(x_sor);
fprintf('Number of Iterations: %d\n', iter_sor);
