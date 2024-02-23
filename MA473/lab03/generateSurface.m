function generateSurface(x, y, data, methodName, dx, dy)
    % Create the surface plot
    figure();
    surf(x, y, data);
    xlabel('Spatial Domain');
    ylabel('Temporal Domain');
    zlabel('Solution Value');
    title('dx = ' + string(dx) + ', dy = ' + string(dy) + ' (' + methodName + ')');
    shading interp;
    colorbar;
end