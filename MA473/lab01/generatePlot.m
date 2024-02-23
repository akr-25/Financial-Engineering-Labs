function generatePlot(spatialDomain, numericalSolution, analyticalSolution, methodName, h, k, id)
    fig = figure('Visible', 'off'); 
    plot(spatialDomain, numericalSolution, spatialDomain, analyticalSolution);
    title('h = ' + string(h) + ' k = ' + string(k) + ' ' + methodName); 
    saveas(fig, [methodName, num2str(id), '.png']);
    close(fig);
end
