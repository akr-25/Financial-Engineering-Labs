% Function for Jacobi method
function [x, iteration] = jacobi_method(A, b, tol, max_iter)
    n = length(b);
    x = zeros(n, 1); % initial guess
    x_new = x;
    iteration = 0;
    while iteration < max_iter
        for i = 1:n
            temp_sum = A(i,:)*x - A(i,i)*x(i);
            x_new(i) = (b(i) - temp_sum) / A(i,i);
        end
        if norm(x_new - x, inf) < tol
            break;
        end
        x = x_new;
        iteration = iteration + 1;
    end
end
