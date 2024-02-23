function gamma = get_gamma(U, dx)

    [~, m] = size(U); 
    gamma = zeros(1, m-2); 
    
    for i = 2:m-1 
        gamma(1, i-1) = (U(end, i+1) - 2*U(end, i) + U(end, i-1))/ (dx*dx); 
    end
end
