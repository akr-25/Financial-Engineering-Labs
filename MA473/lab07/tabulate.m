function tabulate(N, error, delta_x, delta_t, idx)
    % This function is used to tabulate the error and the order of convergence
    % for each different N values
    
    schemes = [string('BTCS'), string('Crank-Nicolson')];
    fprintf('\n\n********  %s Scheme  ********\n', schemes(idx));
    
    T = table((1 : length(N))', N', delta_x', delta_t', (delta_x/2)', (delta_t/2)', error');
    T.Properties.VariableNames = {'IDX', 'N', 'dx', 'dt', 'dx_2', 'dt_2', 'Max_error'};
    disp(T);
    
    figure();
    plot(N, error);
    
    title('Error ( E_{N} ) vs N: ' + schemes(idx) + ' scheme');
    xlabel('N');
    ylabel('Error ( E_{N} )');
end

