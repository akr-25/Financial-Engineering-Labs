function greek_delta = get_delta(U, dx)
    [~, m] = size(U); 
    greek_delta = zeros(1, m-2); 
    
    for i = 2:m-1 
        greek_delta(1, i-1) = (U(end, i+1) - U(end, i-1))/ (2*dx); 
    end
end

