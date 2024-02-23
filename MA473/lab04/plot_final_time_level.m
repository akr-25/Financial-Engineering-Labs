function plot_final_time_level(x, t, U, idx)
    % This function plots the greeks for the corresponding inputs.
    % INPUT: x, y, z (representing the data points)
    % OUTPUT: final time level plot
        
    schemes = [string('FTCS'), string('BTCS'), string('Crank-Nicolson')];
    figure(); 
    hold on; 
    m = sum(x <= 0.5);
    plot(x(1:m), U(1,1:m), 'LineWidth', 1.5);
    plot(x(1:m), U(end,1:m), 'LineWidth', 1.5);
    hold off; 
    legend('t = 0', 't = T'); 
    title('Plot of solutions at final time level using '+ schemes(idx));
%     subtitle(schemes(idx) + ' scheme' + m);
    xlabel('x');
    ylabel('t');
end
