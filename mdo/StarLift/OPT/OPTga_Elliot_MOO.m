% Implement GA for Q3
% Adapted from
% dWo, 3/15/2002
clear 

close all

libraryPathMDOrun = '/Users/giulianac.hofheins/Library/CloudStorage/GoogleDrive-gch72@cornell.edu/My Drive/MDO-Fall 2023/V4 code (11 06 2023)/MDOtoolbox/MDOrun';
libraryPathGA  = '/Users/giulianac.hofheins/Library/CloudStorage/GoogleDrive-gch72@cornell.edu/My Drive/MDO-Fall 2023/V4 code (11 06 2023)/MDOtoolbox/optimizationAlgorithms/GA';
libraryPathmodules = '/Users/giulianac.hofheins/Library/CloudStorage/GoogleDrive-gch72@cornell.edu/My Drive/MDO-Fall 2023/V4 code (11 06 2023)/MDOtoolbox/modules';
addpath(libraryPathMDOrun);
addpath(libraryPathGA);
addpath(libraryPathmodules)

dataSaveFlag = 1;

%bounds on continuous design variables/factors
% variable 1
thrusterSpace = [1,2];

% variable 2
HallThruster_min = 500; 
HallThruster_max = 30000; 
IonThruster_min = 500; 
IonThruster_max = 30000; 

%variable3
propellantSpace = [1,3];

% variable 4
Insul_thk_min = 1e-3; % [m]
Insul_thk_max = 20*1e-3; % [m]

% variable 5
FinishType_min = .2;
FinishType_max = 1;

% variable 6
DebrisRemoved_min = 1;
DebrisRemoved_max = 22;

numDesignVars = 6;

tic

deltaX = 0.00001;
NG= 20; %num generations
BITS_hall_thruster=ceil(log((HallThruster_max-HallThruster_min)/deltaX)/log(2));
BITS_ion_thruster=ceil(log((IonThruster_max-IonThruster_min)/deltaX)/log(2));
BITS_insul_thk = ceil(log((Insul_thk_max-Insul_thk_min)/deltaX)/log(2));
BITS_finish = ceil(log((FinishType_max-FinishType_min)/deltaX)/log(2));
BITS_debrisremoved = ceil(log((DebrisRemoved_max-DebrisRemoved_min)/deltaX)/log(2));
BITS = max([BITS_insul_thk,BITS_ion_thruster,BITS_hall_thruster,BITS_finish]);
mrate=0.001;   %mutation rates
N=100;   % population size

lambdaList = 0:.0003:1;
lambdaInd = 0;
tic
Xp = zeros(N,numDesignVars,NG,length(lambdaList));
Xg = zeros(N,numDesignVars*BITS,NG,length(lambdaList));
for p = 1:length(lambdaList)
lambda = lambdaList(p);
%toc
lambdaInd = lambdaInd+1;

Xp(:,1,1,lambdaInd)=ceil(rand(N,1)+0.5);
Xp(:,2,1,lambdaInd)=(HallThruster_max-HallThruster_min)*(rand(N,1))+HallThruster_min;
Xp(:,3,1,lambdaInd)=ceil(rand(N,1)*3); %prop type
Xp(:,4,1,lambdaInd)=(Insul_thk_max-Insul_thk_min)*(rand(N,1))+Insul_thk_min;
Xp(:,5,1,lambdaInd)=FinishType_max*(rand(N,1))+FinishType_min;
Xp(:,6,1,lambdaInd)=DebrisRemoved_max*(rand(N,1))+DebrisRemoved_min;

% encode initial population
    
for ind = 1: N
    X1g = GA_encode(squeeze(Xp(ind,1,1,lambdaInd)),1,2,BITS);
    %X1g(2:NG,:) = [];
    X2g = GA_encode(squeeze(Xp(ind,2,1,lambdaInd)),HallThruster_min,HallThruster_max,BITS);
    %X2g(2:NG,:) = [];
    X3g = GA_encode(squeeze(Xp(ind,3,1,lambdaInd)),1,3,BITS);
    X4g = GA_encode(squeeze(Xp(ind,4,1,lambdaInd)),Insul_thk_min,Insul_thk_max,BITS);
    %X3g(2:NG,:) = [];
    X5g = GA_encode(squeeze(Xp(ind,5,1,lambdaInd)),FinishType_min,FinishType_max,BITS);
   % X4g(2:NG,:) = [];
    X6g = GA_encode(squeeze(Xp(ind,6,1,lambdaInd)),DebrisRemoved_min,DebrisRemoved_max,BITS);
    %X5g(2:NG,:) = [];
    Xg(ind,:,1,lambdaInd)=[X1g, X2g, X3g, X4g, X5g, X6g];
end

% evaluate initial population

for ind = 1:N
    F(ind,1,lambdaInd)= -MDOrun_MOO(Xp(ind,1,1),Xp(ind,2,1),Xp(ind,3,1),Xp(ind,4,1),Xp(ind,5,1),Xp(ind,6,1),lambda);
end

Favg(1,lambdaInd)=mean(F(:,1));
gen=1;
% plot initial population


% [Xs,Ys,Zs] = meshgrid(-5:deltaX*5:5,-5:deltaX*5:5,-5:deltaX*5:5);
% F_brute_force = Q3GA(Xs,Ys,Zs); 
% [minX,indX] = min(F_brute_force);
% [minY,indY] = min(minX);
% [minZ,indZ] = min(minY);
%[indX,indY,indZ]



%disp('Look at initial population - and type return to start')

%keyboard


    

while gen<NG
    
    % selection - use Roulette Wheel Selection scheme
    
    R=max(F(:,gen,lambdaInd))-min(F(:,gen,lambdaInd));
    for ind=1:N
    FS(ind)=(F(ind,gen,lambdaInd)-min(F(:,gen,lambdaInd)))/R;
    FS(ind)=FS(ind).^2;
    end
    
   
    Xkeep=[]; ind=1;
   while size(Xkeep,1)<N
%     trial=rand;
%     if FS(ind)>trial
      Xkeep=[Xkeep ; squeeze(Xg(ind,:,gen,lambdaInd))];
%     else
%     sel=0;
%     end
    
    ind=ind+1;
    ind=mod(ind,N);
    if ind==0
        ind=1;
    end
    end
    
    
    % crossover - use provided crossover operator
    
    [Xgn]=GA_xover(Xkeep,0.95);
    
    % mutate
    
    [Xgnm]=GA_mutate(Xgn,mrate);
    
    % decode new population
    
    for ind=1:N
        Xpn1=round(GA_decode(Xgnm(ind,1:BITS),1,2,BITS));
        Xpn2=GA_decode(Xgnm(ind,BITS+1:2*BITS),HallThruster_min,HallThruster_max,BITS);
        Xpn3=round(GA_decode(Xgnm(ind,1:BITS),1,3,BITS));
        Xpn4=GA_decode(Xgnm(ind,2*BITS+1:3*BITS),Insul_thk_min,Insul_thk_max,BITS);
        Xpn5=GA_decode(Xgnm(ind,3*BITS+1:4*BITS),FinishType_min,FinishType_max,BITS);
        Xpn6=GA_decode(Xgnm(ind,4*BITS+1:5*BITS),DebrisRemoved_min,DebrisRemoved_max,BITS);
        Xpn(ind,:)=[Xpn1 Xpn2 Xpn3 Xpn4 Xpn5 Xpn6];
    end


    % insert new population
    
    gen=gen+1;
    
    Xg(:,:,gen,lambdaInd)=Xgnm;
    Xp(:,:,gen,lambdaInd)=Xpn;
    
    
    % Compute Fitness of new population
    
    for ind = 1:N
        thrusterCurr = Xp(ind,1,gen,lambdaInd);
        powerCurr = Xp(ind,2,gen,lambdaInd);
        if thrusterCurr == 1
            if powerCurr > HallThruster_max
                powerCurr = HallThruster_max;
                disp('Fixed Power')
            end
            if powerCurr < HallThruster_min
                powerCurr = HallThruster_min;
                disp('Fixed Power')
            end
        elseif thrusterCurr == 2
            if powerCurr > IonThruster_max
                powerCurr = IonThruster_max;
                disp('Fixed Power')
            end
            if powerCurr < IonThruster_min
                powerCurr = IonThruster_min;
                disp('Fixed Power')
            end
        end
        insulCurr = Xp(ind,4,gen,lambdaInd);
        if insulCurr < Insul_thk_min
            insulCurr = Insul_thk_min;
            disp('Fixed Insul Thk')
        end
        debrisRemovedCurr = Xp(ind,6,gen);
        if debrisRemovedCurr < DebrisRemoved_min
            debrisRemovedCurr = DebrisRemoved_min;
            disp('Fixed Debris Removed')
        end
        propellantCurr = Xp(ind, 3, gen, lambdaInd);
        finishCurr = Xp(ind,5,gen,lambdaInd);
   
        F(ind,gen,lambdaInd) = -MDOrun_MOO(thrusterCurr,powerCurr,propellantCurr, insulCurr,finishCurr,debrisRemovedCurr,lambda);
        if abs(F(ind,gen,lambdaInd)) == inf
            disp('is inf')
        end
    end
  
    Favg(gen,lambdaInd)=mean(F(:,gen,lambdaInd));
    
    
    
end
end
disp('Finished GA')
toc
%% plotting 

Favg = abs(Favg);
%maxFitnessesGen = zeros(1,NG,3);
maxFitnessesLam = zeros(length(lambdaList),1);
maxFitnessesLamInd = maxFitnessesLam;

for k = 1:length(lambdaList)
    [maxFit,maxInd] = max(F(:,NG,k));
    maxFitnessesLam(k) = maxFit;
    maxFitnessesLamInd(k) = maxInd;
end
maxFitnessesLam = abs(maxFitnessesLam);

bestCosts = zeros(1,length(lambdaList));
bestTimes = zeros(1,length(lambdaList));
bestVarsArray = zeros(length(lambdaList), numDesignVars);  % Create a 2D array to store bestVars

for k = 1:length(lambdaList)
    bestVars =  Xp(maxFitnessesLamInd(k),:,NG,k);
    [~,bestCost,bestTime,~,~] = MDOrun_MOO(bestVars(1),bestVars(2),bestVars(3),bestVars(4),bestVars(5),bestVars(6),lambdaList(k));
    bestCosts(k)=bestCost/1e6;
    bestTimes(k)=bestTime/3600/24/365;
    % Store bestVars in the 2D array
    bestVarsArray(k, :) = bestVars;

    %results{gen, k} = struct('BestVars', bestVars, 'BestCost', bestCost, 'BestTime', bestTime);
end

%% valid 
validDesigns = bestCosts < 2e3 & bestTimes < 15;

fbestCosts = bestCosts(validDesigns);
fbestTimes = bestTimes(validDesigns);
fbestVarsArray = bestVarsArray(validDesigns, :);

%% plot 
figure(1)
clf(1)
hold on
figureWidth = 800;  % Adjust as needed
figureHeight = 600;  % Adjust as needed
% Set the figure position and size
set(gcf, 'Position', [100, 100, figureWidth, figureHeight]);

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% Define colors for the two cases
b = [3/255,143/255,245/255];
g = [0.4660 0.6740 0.1880];
r = [0.6350 0.0780 0.1840];
for i = 1:length(fbestTimes)
    if fbestVarsArray(i, 1) == 1 
        if fbestVarsArray(i,3) == 1
            plot(fbestCosts(i), fbestTimes(i), 'o','Color', b, 'MarkerSize', 6, 'LineWidth', 2)
        elseif fbestVarsArray(i,3) == 2
            plot(fbestCosts(i), fbestTimes(i), 'o','Color', g, 'MarkerSize', 6, 'LineWidth', 2)
        elseif bestVarsArray(i,3) == 3
            plot(fbestCosts(i), fbestTimes(i), 'o','Color', r,'MarkerSize', 6, 'LineWidth', 2)
        end
    else
        if fbestVarsArray(i,3) == 1
            plot(fbestCosts(i), fbestTimes(i), 'v','Color', b, 'MarkerSize', 6, 'LineWidth', 2)
        elseif fbestVarsArray(i,3) == 2
            plot(fbestCosts(i), fbestTimes(i), 'v','Color', g, 'MarkerSize', 6, 'LineWidth', 2)
        elseif fbestVarsArray(i,3) == 3
            plot(fbestCosts(i), fbestTimes(i), 'v','Color', r, 'MarkerSize', 6, 'LineWidth', 2)
        end
    end
end
ax = gca; % Get the current axes
set(gcf, 'color', 'w')
ax.FontSize = 18; % 14 is the desired font size for tick labels
xlabel('\textbf{Life Cycle Cost (\$ USD million)}', 'FontSize', 22)
ylabel('\textbf{Time (years)}', 'FontSize',22)
title(ax, '\textbf{Multiobjective Design Space}', 'FontSize',26, 'Interpreter', 'latex')

% Add legend
legend('Ion Engine', 'Hall Thruster',...
       'Location', 'NorthEast', 'FontSize', 14);

box on
grid on
%% save
currentFolder = pwd;
currentTime = string(datetime("now"));
currentTime = strrep(currentTime,' ','_');
currentTime = strrep(currentTime,':','_');
currentTime = strrep(currentTime,'-','_');
fileNameAndTarget = strcat(currentFolder,'/','OPTga_MOO_results/MOOdataFINALEXPreal_',currentTime);
if dataSaveFlag == 1
    save(fileNameAndTarget)
end


%% plot 2 (this is just for getting the legend haha)
figure(4);

% First plot with three lines
x_values = linspace(0, 10, 100);
y_values_b = sin(x_values);
y_values_g = cos(x_values);
y_values_r = tan(x_values);
y_values_k = sinh(x_values);

figureWidth = 800;  % Adjust as needed
figureHeight = 600;  % Adjust as needed
set(gcf, 'Position', [100, 100, figureWidth, figureHeight]);

% Second plot with two points
x_points = [3, 7];
y_points = [1, -1];

plot(x_points(1), y_points(1), 'o', 'MarkerEdgeColor', 'k', 'MarkerSize', 10);
hold on;
plot(x_points(2), y_points(2), 'v', 'MarkerEdgeColor', 'k', 'MarkerSize', 10);

plot(x_values, y_values_b, 'Color', [3/255,143/255,245/255], 'LineWidth', 2);

plot(x_values, y_values_g, 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2);
plot(x_values, y_values_r, 'Color', [0.6350 0.0780 0.1840], 'LineWidth', 2);
plot(x_values, y_values_k, 'LineWidth',2, 'Color', 'k')

xlabel('\textbf{X-axis}', 'FontSize', 14);
ylabel('\textbf{Y-axis}', 'FontSize', 14);
title('\textbf{Three Lines}', 'FontSize', 16, 'Interpreter', 'latex');


% Add legend for the first plot
legend('Hall Thruster', 'Ion Engine', 'Xenon', 'Krypton', 'Argon', 'Pareto Front',  'Location', 'Best', 'FontSize', 20);

% Add legend for the second plot
%legend('\bf{Blue Circle}', '\bf{Red Triangle}', 'Location', 'Best');

hold off;

%% legend part two
x_points = [3, 7];
y_points = [1, -1];

figure(4)
plot(x_points(1), y_points(1), 'o','Color', b, 'MarkerSize', 6, 'LineWidth', 2)
legend('Hall Thruster, Xenon', 'Location', 'Best', 'FontSize', 20)
hold on;


% %plot(Favg,LineWidth=1.5)
% hold on
% plot(-maxFitnesses,LineWidth=1.5)
% %plot(-minFitnesses,LineWidth=1.5)
% plot(NG,-maxFitnesses(NG),'ok',LineWidth=3)
% set(gcf, 'color','w')
% box on
% grid on
% xlabel('Generation')
% ylabel('Cost')
% titleStr = sprintf('System Cost over Generations of Genetic Algorithm\n Num Gens = %1.0f, deltaX = %.5f, Mutation Rate = %1.4f Pop Size = %1.0f',NG,deltaX,mrate,N);
% title(titleStr)
% legend('Minimum Cost',...
%    sprintf('Minimum Cost of Final Generation = $%1.4f million',-maxFitnesses(NG)/1e6));
% 
% %  'Function Maximum Value',... 'Function Average Value',
% [bestFitness,indBestFitness] = min(-F(:,NG));
% 
% BestProps = Xp(indBestFitness,:,NG)
% [bestCost,bestTime,bestMass] = MDOrun(BestProps(1),BestProps(2),BestProps(3),BestProps(4),BestProps(5))
% 





