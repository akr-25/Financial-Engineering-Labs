% Function for Gauss-Seidel method
function [x, iteration] = gauss_seidel_method(A, b, tol, max_iter)
    n = length(b);
    x = zeros(n, 1); % initial guess
    iteration = 0;
    while iteration < max_iter
        x_old = x;
        for i = 1:n
            temp_sum = A(i,:)*x - A(i,i)*x(i);
            x(i) = (b(i) - temp_sum) / A(i,i);
        end
        if norm(x - x_old, inf) < tol
            break;
        end
        iteration = iteration + 1;
    end
end
