function [solutionMatrix] = BTCS(hStep, timeStep, xPoints, tPoints, inputFunc, initialF, boundaryFunc1, boundaryFunc2)
% implicitTimeStepping - Execute the implicit method for time-stepping
% @param - hStep, timeStep, xPoints, tPoints, inputFunc, initialF, boundaryFunc1, boundaryFunc2
% @returns - solutionMatrix

    coefficient = timeStep / hStep^2;
    
    solutionMatrix = zeros(tPoints+1, xPoints+1);

    solutionMatrix(1, 1:end) = initialF((0:xPoints)*hStep);
    solutionMatrix(1:end, 1) = boundaryFunc1((0:tPoints)*timeStep);
    solutionMatrix(1:end, end) = boundaryFunc2((0:tPoints)*timeStep);

    % Initialization of coeffMatrix
    coeffMatrix = zeros(xPoints+1, xPoints+1);
    coeffMatrix(1:xPoints+2:end) = 1 + 2*coefficient;
    coeffMatrix(2:xPoints+2:end) = -coefficient;
    coeffMatrix(xPoints+2:xPoints+2:end) = -coefficient;

    coeffMatrix(1,1) = 1;
    coeffMatrix(1,2) = 0;
    coeffMatrix(xPoints+1,xPoints+1) = 1;
    coeffMatrix(xPoints+1,xPoints) = 0;


    for i = 2:tPoints+1
        resultVector = zeros(xPoints+1, 1);
        resultVector(2:xPoints) = solutionMatrix(i-1,2:xPoints) + timeStep*inputFunc((1:xPoints-1)*hStep,(i-1)*timeStep);
        resultVector(1) = solutionMatrix(i,1);
        resultVector(end) = solutionMatrix(i,end);

        solutionMatrix(i,:) = (coeffMatrix\resultVector)';
    end
end
