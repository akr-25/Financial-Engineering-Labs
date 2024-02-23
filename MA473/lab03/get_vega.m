function vega = get_vega(U_plus, U_minus, del_sigma)
    [~, m] = size(U_plus); 
    vega = zeros(1, m-2); 
    
    for i = 2:m-1 
        vega(1, i-1) = (U_plus(end, i) - U_minus(end, i))/ (2*del_sigma); 
    end
    
end
