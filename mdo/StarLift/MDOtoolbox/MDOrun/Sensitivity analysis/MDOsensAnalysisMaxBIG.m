clear all
clc
close all

%% Add paths
libraryPathMDOrun = 'G:\My Drive\College\Grad school\ME Aerospace\Project\V4 code (11 06 2023)\MDOtoolbox\MDOrun';
libraryPathGA  = 'G:\My Drive\College\Grad school\ME Aerospace\Project\V4 code (11 06 2023)\MDOtoolbox\optimizationAlgorithms\GA';
libraryPathmodules = 'G:\My Drive\College\Grad school\ME Aerospace\Project\V4 code (11 06 2023)\MDOtoolbox\modules';
addpath(libraryPathMDOrun);
addpath(libraryPathGA);
addpath(libraryPathmodules);

%% Create step size for each parameter
%Reference for syntax
% [BestProps, bestCost,bestTime,bestMass, bestMassFractions, bestMassFractionsSubsys, bestCostFractions, bestCostFractionsSubsys, bestCostFractionsSC, bestPropFtdrag] = OPTgaFXN()
[bestProps, costStar ,~,~, ~, ~, ~, ~, ~, ~] = OPTgaFXNBIG(); % costStar is J*, or cost using optimal parameters
bestParams = MDOgenerateParamater();

%corresponding step sizes for each. using step size = 0.01% of x0, as per Haji 2023 lecture
p = 0.0001; %scale factor based on above definition

paramFields = fieldnames(bestParams);
numericParams = zeros(length(paramFields),1); %convert struct to numeric array


for i = 1:length(paramFields)
    if length(bestParams.(paramFields{i})) == 1
        numericParams(i) = bestParams.(paramFields{i}); %only add the numbers, not the array
    end
end
problemParams = [1 2 3 4 5 6 7 8 9 10 11 17 18 19 22 35 36 42 46 47 48]; %parameters that do not make sense to vary

for i = 1:length(problemParams)
    numericParams(problemParams(i)) = 0;
end
paramStep = p * numericParams;

%% uncomment at most one of the following two lines
% paramStep(42) = 1; %uncomment this line if you want to vary number of debris by integer
paramStep(42) = 0; %uncomment this line if you don't want to vary the number of debris at all

%%
paramStep(51) = 0; %this is a robotic mission, no variance to it
paramStep(52) = 1; %vary by integer year
paramStep(53) = 0; %this is a new mission, no variance to it
paramStep(54) = p; %give difficulty a nonzero step
%%
posSens = sensGeneratorBIG(paramStep,costStar,bestProps,numericParams);
negSens = sensGeneratorBIG((paramStep)*-1,costStar,bestProps,numericParams); %take a negative step
labels = ["Radius of Earth", "Disposal Radius", "Radius of debris", "Launch radius", "Mass of single debris",...
          "$\mu_E$", "$g_0$", "Molar mass of Xe", "Molar mass of Kr", "Molar mass of Ar",...
          "Solar incidence at Earth", "Solar panel efficiency", "Power factor of safety", "Solar panel area density", "Battery density",...
          "Battery minimum charge", "Debris body diameter", "Debris body length", "Debris surface area", "Atmospheric density of Earth at 200km altitude",...
          "Drag coefficient at disposal radius", "Orbital velocity at disposal radius", "Spacecraft density", "Mass of collection system", "Price of Xenon",...
          "Cost of collection system", "Spacecraft bus radius", "something", "something else", "Thermal conductivity of MLI",...
          "Earth emitted IR flux (lower boundary)", "Earth emitted IR flux (higher boundary)", "Earth albedo (lower boundary)", "Earth albedo (higher boundary)", "boltzmann constant",...
          "Solar incidence flux (lower boundary)", "Solar incidence flux (higher boundary)", "Absorptance of finish (diffuse quartz)", "Emittance of finish (diffuse quartz)", "Minimum allowable electronics temperature",...
          "Maximum allowable electronics temperature", "Total debris", "R_univ", "Propellant tank temperature", "Propellant tank material density",...
          "Propellant tank yield strength", "MWxe", "MWkr", "MWar", "Number of satellites",...
          "Mission type", "Launch year", "B", "Mission difficulty", "Cost of using Falcon 9 to achieve SSO",...
          "General cost of using Falcon 9", "Cost of using Falcon Heavy to achieve SSO", "General cost of using Falcon Heavy", "Operational cost", "Upfront cost"];
newPosSens = [];
newNegSens = [];
newLabels = "";
j = 1;

for i = 1:length(posSens) %getting rid of NaNs
    if ~(isnan(posSens(i))) && (posSens(i) ~= 0)
        newPosSens(j) = posSens(i);
        newNegSens(j) = negSens(i);
        newLabels(j) = labels(i);
        j = j + 1;
    end
end

%%

%% Plot generation
sensChart = zeros(length(newPosSens), 2);
for s = 1:length(newPosSens)
    sensChart(s, 1) = -newNegSens(s);
    sensChart(s, 2) = newPosSens(s);
end

x = categorical(newLabels);
x = reordercats(x,newLabels);

set(groot,'defaultAxesTickLabelInterpreter','latex')
barh(x,sensChart, 'stacked', 'BaseValue',0)
title('\bf{Sensitivity of cost model based on full parameter space}', 'Interpreter','latex')
xlabel('\bf{Normalized sensitivities}','Interpreter','latex')
ylabel('\bf{Parameters}','Interpreter','latex')
yticklabels(newLabels)
% set(gca,'yticklabel',labels)
legend({'Negative step','Positive step'})

sensMultiPlot(sensChart, newLabels)
