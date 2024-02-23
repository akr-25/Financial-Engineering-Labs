function theta = get_theta(U, dt)
   
    [n, m] = size(U); 
    theta = zeros(1, m-2); 
    
    for i = 2:m-1 
        theta(1, i-1) = (U(n, i) - U(n-1, i))/ (dt); 
    end
end

