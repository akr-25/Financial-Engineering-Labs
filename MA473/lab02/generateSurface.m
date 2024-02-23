function generateSurface(x, y, data, methodName, h, k, id)
    % Create a figure with visibility set to 'off'
    fig = figure('Visible', 'off'); 
    
    % Create the surface plot
    surf(x, y, data);
    xlabel('Spatial Domain');
    ylabel('Temporal Domain');
    zlabel('Solution Value');
    title('h = ' + string(h) + ', k = ' + string(k) + ' (' + methodName + ')');
    shading interp;
    colorbar;
    
    % Save the figure
    saveas(fig, [methodName, '_surface_', num2str(id), '.png']);
    
    % Close the figure after saving
    close(fig);
end
