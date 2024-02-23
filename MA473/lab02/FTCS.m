function [solutionGrid] = FTCS(spatialStep, timeStep, spatialPoints, timePoints, externalFunc, initialCondition, leftBoundaryFunc, rightBoundaryFunc, startX)
  
    lambda = timeStep / spatialStep^2;
    
    solutionGrid = zeros(timePoints+1, spatialPoints+1);
    solutionGrid(1, :) = initialCondition(startX + (0:spatialPoints)*spatialStep);
    
    for currTimeIdx = 1:timePoints+1
        currTime = (currTimeIdx-1) * timeStep;
        solutionGrid(currTimeIdx, 1) = leftBoundaryFunc(startX, currTime);
        solutionGrid(currTimeIdx, end) = rightBoundaryFunc(startX + spatialPoints*spatialStep, currTime);
    end
    
    for currTimeIdx = 2:timePoints+1
        for currSpaceIdx = 2:spatialPoints
            prevTime = (currTimeIdx-1) * timeStep;
            currPosition = startX + (currSpaceIdx-1) * spatialStep;
            solutionGrid(currTimeIdx, currSpaceIdx) = lambda * solutionGrid(currTimeIdx-1, currSpaceIdx-1) + (1-2*lambda) * solutionGrid(currTimeIdx-1, currSpaceIdx) + lambda * solutionGrid(currTimeIdx-1, currSpaceIdx+1) + timeStep * externalFunc(currPosition, prevTime);
        end
    end
end
