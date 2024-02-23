function rho = get_rho(U_plus, U_minus, del_r)
    
    [~, m] = size(U_plus); 
    rho = zeros(1, m-2); 
    
    for i = 2:m-1 
        rho(1, i-1) = (U_plus(end, i) - U_minus(end, i))/ (2*del_r); 
    end
    
end
