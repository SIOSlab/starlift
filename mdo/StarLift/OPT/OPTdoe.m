%% Design of Experiments Code
% Full Factorial Approach
% l^n number of experiments, l = levels of n = factors(DV)
% Design Variables = factors
clear
close all
clc

% Specify the path to your library folder
libraryPath = '/Users/giulianac.hofheins/Library/CloudStorage/GoogleDrive-gch72@cornell.edu/My Drive/MDO-Fall 2023/V4 code (11 06 2023)/MDOtoolbox/MDOrun';
libraryPath2 = '/Users/giulianac.hofheins/Library/CloudStorage/GoogleDrive-gch72@cornell.edu/My Drive/MDO-Fall 2023/V4 code (11 06 2023)/MDOtoolbox/modules';
% Add the library path to MATLAB's search path
addpath(libraryPath);
addpath(libraryPath2);

tic
% Design Variables
n1 = 2; %thruster type (Discrete, 1 = Hall, 2 = Ion)
n2 = 4; %power to propulsion (Continuous)
n3 = 3; %propellant type (Discrete, 1 = xenon, 2 = krypton, 3 = argon)
n4 = 4; %insulation thickness (Continuous)
n5 = 4; %surface finish (Continuous)
n6 = 4; %# debris removed (Discrete)
dVnumber = 6;

%number of experiments
nexp = n1*n2*n3*n4*n5*n6; 

%bounds on continuous design variables/factors
HallThruster_min = 500;
HallThruster_max = 30000;
IonThruster_min = 500;
IonThruster_max = 30000;
Insul_thk_min = 1e-3; % [m]
Insul_thk_max = 20*1e-3; % [m]
Surface_finish_min = .2;
Surface_finish_max = .8;

%creation of full design space
thrusterSpace   = 1:n1; %hall and gridded ion
propellantSpace = 1:n3; %xenon, krypton, hall
tauLinSpace = linspace(Insul_thk_min,Insul_thk_max,n4); %continuous -> discrete
surfaceFinishSpace = linspace(Surface_finish_min, Surface_finish_max, n5); %continuous -> discrete
debrisRemovedSpace = [1 5 11 22];  %1-n debris removed


%%initialize output arrays
exp = zeros(nexp,dVnumber); %experiments, the level of each factor for full factorial

%results = objective/constraint output at each experiment
    %columns = cost, time, mass; rows = exp #
results = zeros(nexp,3);

%seperated output lists
timelist = zeros(nexp,1);
costlist = zeros(nexp,1);
masslist = zeros(nexp,1);

%levels of DV initialization
levels =  zeros(nexp,dVnumber); 


%counter variable
%for loop
    %calculates output vector for each # of levels for each factor
l = 1;
    for i = 1:n1
       switch thrusterSpace(i)
        case 1
            ptpLinSpace = linspace(HallThruster_min, HallThruster_max, n2);
        case 2
            ptpLinSpace = linspace(IonThruster_min, IonThruster_max, n2);
       end
       for j = 1:n2
            for k = 1:n3
                for z = 1:n4
                    for ii = 1:n5
                        for jj = 1:n6
    
                            %exp array = numerical level of the factors at each exp
                            exp(l,:) = [i,j,k,z,ii,jj];
                
                            %extract levels of factors
                            thrustertype  = thrusterSpace(i);
                            ptp = ptpLinSpace(j);
                            propellanttype = propellantSpace(k);
                            tau = tauLinSpace(z);
                            surfaceFinish = surfaceFinishSpace(ii);
                            debrisRemoved = debrisRemovedSpace(jj);
                            levels(l,:) = [thrustertype, ptp, propellanttype, tau, surfaceFinish, debrisRemoved];
                
                            %run model + objective/constraint output
                            [cost,time, mass] = MDOrun(thrustertype, ptp,propellanttype, tau, surfaceFinish, debrisRemoved);
                
                            %sanity check
                            cost = double(cost);
                            if imag(cost)>0
                                cost = NaN;
                            end
                
                            %results array
                            results(l,:) =  [cost, time, mass];
                            costlist(l) = cost;
                            timelist(l) = time;
                            masslist(l) = mass;
                     
                            %increase counter variable
                            l = l+1;
                        end
                    end
                end
            end
       end
    end

 % Filter out infeasible solutions
feasibleIdx = costlist <= 2200e6; % You can adjust this threshold as needed

costlist = costlist(feasibleIdx);
timelist = timelist(feasibleIdx);
masslist = masslist(feasibleIdx);
results = results(feasibleIdx, :);
exp = exp(feasibleIdx,:);
levels = levels(feasibleIdx,:);



%% effects calculations

%initialize arrays
maxn = max([n1,n2,n3, n4, n5, n6]);
Aeffects = zeros(maxn, 1);
Beffects = zeros(maxn, 1);
Ceffects = zeros(maxn, 1);
Deffects = zeros(maxn, 1);
Eeffects = zeros(maxn, 1);
Feffects = zeros(maxn, 1);
effects = [];

%extract experiments with same factor/level (all exp with A1, A2, etc)
%DVeffects=effect of factor/level in dollars
Amax = 0;
Bmax = 0;
Cmax = 0; 
Dmax = 0;
Emax = 0;
Fmax = 0;

for Aindex = 1:n1
    expAsubset = find(exp(:,1) == Aindex);
    Aeffects(Aindex) = mean(costlist) - mean(costlist(expAsubset));
end

for Bindex = 1:n2
    expBsubset = find(exp(:,2) == Bindex);
    Beffects(Bindex) = mean(costlist) - mean(costlist(expBsubset));
end
% blah = mean(costlist) 
% blahblah = mean(costlist(find(exp(:,2) == 10)))
% blahblahblah = mean(costlist) - mean(costlist(find(exp(:,2) == 10)))

for Cindex = 1:n3
    expCsubset = find(exp(:,3) == Cindex);
    Ceffects(Cindex) = mean(costlist) - mean(costlist(expCsubset));
end

for Dindex = 1:n4 %surface finish
    expDsubset = find(exp(:,4) == Dindex);
    Deffects(Dindex) = mean(costlist) - mean(costlist(expDsubset));
end

for Eindex = 1:n5
    expEsubset = find(exp(:,5) == Eindex);
    Eeffects(Eindex) = mean(costlist) - mean(costlist(expEsubset));
end

for Findex = 1:n6
    expFsubset = find(exp(:,6) == Findex);
    Feffects(Findex) = mean(costlist) - mean(costlist(expFsubset));
end

effects = [effects, [Aeffects, Beffects, Ceffects, Deffects, Eeffects, Feffects]];

format long
fprintf('Max Cost')
max(costlist/1e6)
fprintf('Min Cost')
min(costlist/1e6)
fprintf('Mean Cost')
mean(costlist/1e6)
fprintf('Median Cost')
median(costlist/1e6)
fprintf('Variance')
var(costlist/1e6)
fprintf('Standard Deviation')
std(costlist/1e6)


[valueA, indexA] = max(Aeffects);
xbestA = thrusterSpace(indexA);
[valueB, indexB] = max(Beffects);
xbestB = ptpLinSpace(indexB);
[valueC, indexC] = max(Ceffects);
xbestC = propellantSpace(indexC);
[valueD, indexD] = max(Deffects);
xbestD = tauLinSpace(indexD);
[valueE, indexE] = max(Eeffects);
xbestE = surfaceFinishSpace(indexE);
[valueF, indexF] = max(Feffects);
xbestF = debrisRemovedSpace(indexF);




fprintf('xbest, **power to prop in kW')
xbest = [xbestA, xbestB/1e3, xbestC, xbestD, xbestE, xbestF]'

fprintf('Min Cost')
min(costlist)

[valueCost, indexCost] = min(costlist);
fprintf('Mission time at minimum cost, in years')
timelist(indexCost)
fprintf('System mass at minimum cost, in kg')
masslist(indexCost)

figure(1)
hold on
set(groot,'defaultAxesTickLabelInterpreter','latex');
ax = gca;
ax.FontSize = 22; 
binmin = min(costlist)/1e6;
binsize1 = 1e8/1e6;
binsize2 = 1e6/1e6;
sortedcostlist = sort(costlist);
m = round(0.5 * numel(sortedcostlist));
h = histogram(costlist/1e6,'BinLimits', [binmin, max(costlist)], 'BinWidth', binsize1);
xlabel('\textbf{Cost (\$USD million)}', 'Interpreter', 'latex','FontSize', 22)
ylabel('\textbf{Count}','Interpreter', 'latex','FontSize', 22)
%ax('Interpreter', 'latex')
title('\textbf{Cost Distribution from DOE}', 'Interpreter', 'latex', 'FontSize', 26)

figure(2)
%histogram(sortedcostlist(1:1000), 'BinLimits', [binmin, max(sortedcostlist)],'BinWidth', binsize2)
histogram(sortedcostlist(1:m)/1e6)
xlabel('\textbf{Cost}', 'Interpreter', 'latex','FontSize', 22)
ylabel('\textbf{Count}','Interpreter', 'latex','FontSize', 22)
%ax('Interpreter', 'latex')
title('\textbf{Cost Distribution of bottom \frac{1}{2} from DOE}', 'Interpreter', 'latex', 'FontSize', 26)


disp('Stats of bottom 1/2')
disp('Mean')
mean(sortedcostlist(1:m)/1e6)
disp('Median')
median(sortedcostlist(1:m)/1e6)
disp('Min')
min(sortedcostlist(1:m)/1e6)
disp('Max')
max(sortedcostlist(1:m)/1e6)
disp('Variance')
var(sortedcostlist(1:m)/1e6)
disp('std')
std(sortedcostlist(1:m)/1e6)


% Plotting the first graph
figure(3)
subplot(1, 2, 1);
hold on
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
ax = gca;
ax.FontSize = 22; 
h = histogram(costlist/1e6, 'BinLimits', [binmin, max(costlist)/1e6],'BinWidth', binsize1);
xlabel('\textbf{Cost (\$USD million)}', 'Interpreter', 'latex', 'FontSize', 24)
ylabel('\textbf{Count}', 'Interpreter', 'latex', 'FontSize', 24)
title('\textbf{Cost Distribution from DOE}', 'Interpreter', 'latex', 'FontSize', 26)


% Add a text box for bin size
%text(0.6, 0.9, ['Bin Size: 100e6 million'], 'Units', 'normalized', 'FontSize', 20, 'BackgroundColor', 'w', 'Interpreter', 'Latex');

% Plotting the second graph
% subplot(1, 2, 2); 
% histogram(sortedcostlist(1:m)/1e6, 'BinWidth', 25)
% set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
% ax = gca;
% ax.FontSize = 22;
% title('\textbf{Lower Half of Cost Distribution (b)}', 'Interpreter', 'latex', 'FontSize', 26)
% xlabel('\textbf{Cost (\$USD million)}', 'Interpreter', 'latex', 'FontSize', 22)
% ylabel('\textbf{Count}', 'Interpreter', 'latex', 'FontSize', 22)

% Add a text box for bin size
%text(0.6, 0.9, ['Bin Size: 1e6 million'], 'Units', 'normalized', 'FontSize', 20, 'BackgroundColor', 'w', 'Interpreter', 'latex');

subplot(1,2,2);
propsyseffects = [-46236519.14 ,35728219.34,0,0]/1e6;
powereffects = [0, -64626447.33, 112478029.7, -61475003.68]/1e6;
propellanteffects = [-26834217.99, 12339528.3, 49653833.22, 0]/1e6;
taueffects = [505688.9771, 131412.4539, -169803.6182, -467297.8128]/1e6;
alphaeffects = [4768432.41, 2238010.459, -1331510.444, -5674932.426]/1e6;
ndebriseffects = [-1121474788, -215803600.9, 206173441.7,348961643.4]/1e6;

propsyseffects = effects(:,1)';
powereffects  = [0; effects(2:end, 2)]';
propellanteffects = effects(:,3)';
taueffects = effects(:,4)';
alphaeffects = effects(:, 5)';
ndebriseffects = effects(:,6)';


effectss = -1*[ndebriseffects; propellanteffects; powereffects;propsyseffects]/1e6;
titles = {'$n_{debris}$', '$\zeta$', '$P_{propulsion}$','$k_{propsys}$'};

barh(effectss, 'stacked')
yticklabels(titles)

% Adjust label font size
set(gca, 'FontSize', 24);
xlabel('\textbf{Effect Size (\$USD million)}', 'FontSize',24);
title('\textbf{Main Effects}', 'FontSize', 26);
legend({'Level 1', 'Level 2', 'Level 3', 'Level 4'}, 'Interpreter', 'latex', 'FontSize', 22);
toc


