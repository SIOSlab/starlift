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

%Reference for syntax
% [BestProps, bestCost,bestTime,bestMass, bestMassFractions, bestMassFractionsSubsys, bestCostFractions, bestCostFractionsSubsys, bestCostFractionsSC, bestPropFtdrag] = OPTgaFXN()
[bestProps, costStar ,~,~, ~, ~, ~, ~, ~, ~] = OPTgaFXN(); % costStar is J*, or cost using optimal parameters

%corresponding step sizes for each. using step size = 0.01% of x0, as per Haji 2023 lecture
p = 0.0001; %scale factor based on above definition
propsStep = p * bestProps;

%% step size for propulsion and propellant choice are non-numeric
propsStep(1) = 0;
propsStep(3) = 0;

%step size for number of debris is integerial
propsStep(6) = 1; %comment this line out to let number of debris to be continuous, uncomment to force integer
%%

posSens = sensGenerator(propsStep, bestProps, costStar); %sensitivity for positive step
negSens = sensGenerator((propsStep*-1), bestProps, costStar); %sensitivity for negative step

%% Plot generation
sensChart = zeros(length(posSens), 2);
for s = 1:length(posSens)
    sensChart(s, 1) = -negSens(s);
    sensChart(s, 2) = posSens(s);
end

labels = [sprintf("Thruster type; Default: %d",bestProps(1));...
    sprintf("Propulsion power; Default: %.4g kW",bestProps(2)/1e3);...
    sprintf("Propellant selection; Default: %d",bestProps(3));...
    sprintf("Insulation thickness; Default: %.4g [m]",bestProps(4));...
    sprintf("Surface finish; Default: %.4g",bestProps(5));...
    sprintf("Number of debris removed; Default: %d",round(bestProps(6)))];
barh(sensChart, 'stacked', 'BaseValue',0)
title('\textbf{Sensitivity of cost model to direct variation of design variables}','Interpreter','latex')
xlabel('\textbf{Normalized sensitivities}', 'Interpreter','latex')
ylabel('\textbf{Design variable}','Interpreter','latex')
set(gca,'yticklabel',labels)
legend({'Negative step','Positive step'})
