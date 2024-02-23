% Function for Successive Over-Relaxation (SOR) method
function [x, iteration] = sor_method(A, b, omega, tol, max_iter)
    n = length(b);
    x = zeros(n, 1); % initial guess
    x_new = x;
    iteration = 0;
    while iteration < max_iter
        x_old = x;
        for i = 1:n
            temp_sum = A(i,:)*x - A(i,i)*x(i);
            x(i) = (1-omega)*x(i) + omega*(b(i) - temp_sum) / A(i,i);
        end
        if norm(x - x_old, inf) < tol
            break;
        end
        iteration = iteration + 1;
    end
end