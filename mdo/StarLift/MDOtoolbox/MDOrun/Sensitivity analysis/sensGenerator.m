function normSens = sensGenerator(propsStep,bestProps, costStar)
%SENSGENERATOR generates the normalized sensitivity based on step size and
%experimentally found ideal variables
%   Detailed explanation goes here

newCost = zeros([1, length(propsStep)]); %create list of costs as result of sens. analysis

for i = 1:length(propsStep)
    propsTemp = bestProps;
    propsTemp(i) = bestProps(i) + propsStep(i); %feed in variables with changing one variable at a time
    [cost,~,~,~,~,~,~,~,~] = MDOrun(propsTemp(1),propsTemp(2),propsTemp(3),propsTemp(4),propsTemp(5),propsTemp(6));
    newCost(i) = cost;
end


costSens = (newCost - costStar)/costStar; %normalizing sensitivity to be (dJ/J) / (dx/x)
varSens = propsStep ./ bestProps;
normSens = costSens ./ varSens; %sensitivity due to positive step

%% thruster choice and prop choice are not quantifiable. comment/uncomment this section as you see fit
normSens(1) = 0;
normSens(3) = 0;
%%
end

