errorValues = 500:500:10000;
numPoints = 500:500:10000;

for idx = 1:length(numPoints)
    timeStep = 1/numPoints(idx);
    timePoints = 0 + timeStep : timeStep : 4 - timeStep;
    numIterations = 200;
    errors = zeros(1, numIterations);
    
    for itr = 1:numIterations
        brownianMotion = sqrt(timeStep) * randn(length(timePoints), 1);
        approxSolution = EulerMurayamaLangevin(10, 1, 0, timeStep, timePoints, brownianMotion);
        exactSolution = Langevin(10, 1, 0, timeStep, timePoints, brownianMotion);
        errors(itr) = mean(abs(exactSolution - approxSolution));
    end
    
    errorValues(idx) = mean(errors);
end

loglog(numPoints, errorValues, '-s'), hold on;
loglog(numPoints, 1./sqrt(numPoints)), hold on;
loglog(numPoints, 1./numPoints), hold off;
legend('loglog numPoints vs errorValues', 'loglog numPoints vs 1/sqrt(numPoints)', 'loglog numPoints vs 1/numPoints');

function X = Langevin(mu, sigma, initValue, timeStep, timePoints, brownianMotion)
    X = zeros(size(timePoints));
    numPoints = length(timePoints);
    
    for i = 1:numPoints
        if i == 1
            X(i) = exp(-mu*timeStep)*(initValue + sigma*brownianMotion(i));
        else
            X(i) = exp(-mu*timeStep)*(X(i-1) + sigma*brownianMotion(i));
        end
    end
end

function approxX = EulerMurayamaLangevin(mu, sigma, initValue, timeStep, timePoints, brownianMotion)
    numPoints = length(timePoints);
    approxX = zeros(size(timePoints));
    
    for i = 1:numPoints
        if i == 1
            approxX(i) = initValue - mu*initValue*timeStep + sigma*brownianMotion(i);
        else
            approxX(i) = approxX(i-1) - mu*approxX(i-1)*timeStep + sigma*brownianMotion(i);
        end
    end
end
