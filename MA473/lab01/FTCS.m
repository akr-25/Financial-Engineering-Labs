function [solutionGrid] = FTCS(spatialStep, timeStep, spatialPoints, timePoints, externalFunc, initialCondition, leftBoundaryFunc, rightBoundaryFunc)
  
    lambda = timeStep / spatialStep^2;
    
    solutionGrid = zeros(timePoints+1, spatialPoints+1);
    solutionGrid(1, :) = initialCondition((0:spatialPoints)*spatialStep);
    solutionGrid(:, 1) = leftBoundaryFunc((0:timePoints)*timeStep);
    solutionGrid(:, end) = rightBoundaryFunc((0:timePoints)*timeStep);
    
    for currTimeIdx = 2:timePoints+1
        for currSpaceIdx = 2:spatialPoints
            prevTime = (currTimeIdx-1)*timeStep;
            currPosition = (currSpaceIdx-1)*spatialStep;
            solutionGrid(currTimeIdx, currSpaceIdx) = lambda*solutionGrid(currTimeIdx-1, currSpaceIdx-1) + (1-2*lambda)*solutionGrid(currTimeIdx-1, currSpaceIdx) + lambda*solutionGrid(currTimeIdx-1, currSpaceIdx+1) + timeStep*externalFunc(currPosition, prevTime);
        end
    end
end
