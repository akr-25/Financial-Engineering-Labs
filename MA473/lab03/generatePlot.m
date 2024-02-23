function generatePlot(spatialDomain, numericalSolution, methodName, dx, dt)
    figure();
    plot(spatialDomain, numericalSolution);
    title('dx = ' + string(dx) + ' dt = ' + string(dt) + ' ' + methodName); 
end