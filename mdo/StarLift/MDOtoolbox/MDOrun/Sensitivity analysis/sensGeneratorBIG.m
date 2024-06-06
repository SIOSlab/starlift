function normSens = sensGeneratorBIG(step,costStar,bestProps,params)
%SENSGENERATOR generates the normalized sensitivity based on step size and
%experimentally found ideal variables
%   Detailed explanation goes here

costs = zeros(length(step),1); %create empty vector of costs

for i = 1:length(costs)
    tempStep = zeros(length(step),1);
    tempStep(i) = step(i); %feed in the parameter one step at a time
    tempParam = MDOgenerateParamaterSENS(tempStep);
    [costs(i),~,~, ~, ~, ~, ~, ~, ~] = MDOrunSENS(bestProps(1),bestProps(2),bestProps(3),bestProps(4),bestProps(5),bestProps(6),tempParam);

end


costSens = (costs - costStar)/costStar; %normalizing sensitivity to be (dJ/J) / (dx/x)
varSens = step ./ params;
normSens = costSens ./ varSens; %sensitivity due to positive step


%%
end

