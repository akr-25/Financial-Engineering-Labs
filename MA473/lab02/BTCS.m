function [solutionMatrix] = BTCS(hStep, timeStep, xPoints, tPoints, inputFunc, initialF, boundaryFunc1, boundaryFunc2, startX)
% BTCS - Execute the implicit method for time-stepping
% @param - hStep, timeStep, xPoints, tPoints, inputFunc, initialF, boundaryFunc1, boundaryFunc2, startX
% @returns - solutionMatrix

    coefficient = timeStep / hStep^2;
    solutionMatrix = zeros(tPoints+1, xPoints+1);

    spatialDomain = startX + (0:xPoints) * hStep;
    solutionMatrix(1, :) = initialF(spatialDomain);

    for timeIdx = 1:tPoints+1
        currTime = (timeIdx-1) * timeStep;
        solutionMatrix(timeIdx, 1) = boundaryFunc1(startX, currTime);
        solutionMatrix(timeIdx, end) = boundaryFunc2(startX + xPoints * hStep, currTime);
    end

    % Initialization of coeffMatrix
    coeffMatrix = zeros(xPoints+1, xPoints+1);
    coeffMatrix(1:xPoints+2:end) = 1 + 2*coefficient;
    coeffMatrix(2:xPoints+2:end) = -coefficient;
    coeffMatrix(xPoints+2:xPoints+2:end) = -coefficient;

    coeffMatrix(1,1) = 1;
    coeffMatrix(1,2) = 0;
    coeffMatrix(xPoints+1,xPoints+1) = 1;
    coeffMatrix(xPoints+1,xPoints) = 0;

    for timeIdx = 2:tPoints+1
        resultVector = zeros(xPoints+1, 1);
        currentSpatialPositions = startX + hStep * (1:xPoints-1);
        prevTime = (timeIdx - 2) * timeStep;
        resultVector(2:xPoints) = solutionMatrix(timeIdx-1, 2:xPoints) + timeStep * inputFunc(currentSpatialPositions, prevTime);
        resultVector(1) = solutionMatrix(timeIdx, 1);
        resultVector(end) = solutionMatrix(timeIdx, end);

        solutionMatrix(timeIdx, :) = coeffMatrix \ resultVector;
    end
end
