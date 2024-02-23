function [results] = CN(spatialInterval, timeInterval, numSpatialSteps, numTimeSteps, externalFunction, initialProfile, boundaryAtStart, boundaryAtEnd, startX)
  
    lambda = timeInterval / spatialInterval^2;
    
    results = zeros(numTimeSteps+1, numSpatialSteps+1);
    results(1, :) = initialProfile(startX + [0:numSpatialSteps] * spatialInterval);
    
    for timeStep = 1:numTimeSteps+1
        currTime = (timeStep-1) * timeInterval;
        results(timeStep, 1) = boundaryAtStart(startX, currTime);
        results(timeStep, end) = boundaryAtEnd(startX + numSpatialSteps * spatialInterval, currTime);
    end
    
    systemMatrix = zeros(numSpatialSteps+1, numSpatialSteps+1);
    systemMatrix(1:numSpatialSteps+2:end) = 1 + lambda;
    systemMatrix(2:numSpatialSteps+2:end) = -lambda/2;
    systemMatrix(numSpatialSteps+2:numSpatialSteps+2:end) = -lambda/2;

    systemMatrix(1,1) = 1;
    systemMatrix(1,2) = 0;
    systemMatrix(numSpatialSteps+1,numSpatialSteps+1) = 1;
    systemMatrix(numSpatialSteps+1,numSpatialSteps) = 0;

    for timeStep = 2:numTimeSteps+1
        rhsVector = zeros(numSpatialSteps+1, 1);
        
        currentSpatialPositions = startX + spatialInterval * (1:numSpatialSteps-1);
        previousTime = (timeStep - 2) * timeInterval;
        rhsVector(2:numSpatialSteps) = results(timeStep-1,1:numSpatialSteps-1) * lambda/2 + (1 - lambda) * results(timeStep-1,2:numSpatialSteps) + results(timeStep-1,3:numSpatialSteps+1) * lambda/2 + timeInterval * externalFunction(currentSpatialPositions, previousTime);
        rhsVector(1) = results(timeStep, 1);
        rhsVector(end) = results(timeStep, end);

        results(timeStep, :) = systemMatrix \ rhsVector;
    end
end
