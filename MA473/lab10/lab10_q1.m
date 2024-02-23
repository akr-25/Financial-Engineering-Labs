errorValues = 500:500:10000;
numPoints = 500:500:10000;

for idx = 1:length(numPoints)
    timeStep = 1/numPoints(idx);
    timePoints = 0 + timeStep : timeStep : 1 - timeStep;
    numIterations = 2000;
    errors = zeros(1, numIterations);
    
    for itr = 1:numIterations
        brownianMotion = sqrt(timeStep) * randn(length(timePoints), 1);
        approxSolution = EulerMurayama(0.75, 0.3, 307, timeStep, timePoints, brownianMotion);
        exactSolution = BlackScholes(0.75, 0.3, 307, timeStep, timePoints, brownianMotion);
        errors(itr) = mean(abs(exactSolution - approxSolution));
    end
    
    errorValues(idx) = mean(errors);
end

loglog(numPoints, errorValues, '-s'), hold on;
loglog(numPoints, 1./sqrt(numPoints)), hold off;
legend('loglog numPoints vs errorValues', 'loglog numPoints vs 1/sqrt(numPoints)');

function X = BlackScholes(mu, sigma, initValue, timeStep, timePoints, brownianMotion)
    X = zeros(size(timePoints));
    numPoints = length(timePoints);
    
    for i = 1:numPoints
        if i == 1
            X(i) = initValue * exp((mu - 0.5 * sigma^2) * timeStep + sigma * brownianMotion(i));
        else
            X(i) = X(i - 1) * exp((mu - 0.5 * sigma^2) * timeStep + sigma * brownianMotion(i));
        end
    end
end

function approxX = EulerMurayama(mu, sigma, initValue, timeStep, timePoints, brownianMotion)
    numPoints = length(timePoints);
    approxX = zeros(size(timePoints));
    
    for i = 1:numPoints
        if i == 1
            approxX(i) = initValue + mu * initValue * timeStep + sigma * initValue * brownianMotion(i);
        else
            approxX(i) = approxX(i-1) + mu * approxX(i-1) * timeStep + sigma * approxX(i-1) * brownianMotion(i);
        end
    end
end
