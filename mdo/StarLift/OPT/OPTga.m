% Implement GA for Q3
% Adapted from
% dWo, 3/15/2002
clear 
clc
close all

% CHANGE THIS TO YOUR LOCAL PATH
folderpath = 'G:\My Drive\College\Grad school\ME Aerospace\Project\StarliftMDO\Starlift-MDO';

addpath(genpath(folderpath))

excelflag = 0;

varFlag = 1; % 0 for power, 1 for mass



%bounds on continuous design variables/factors
% variable 1
thrusterSpace = [1,2];

% variable 2
HallThruster_min = 500; 
HallThruster_max = 30000; 
IonThruster_min = 500; 
IonThruster_max = 30000; 

%variable 3
propellantSpace = [1,3];

% variable 4
% Insul_thk_min = 1e-3; % [m]
% Insul_thk_max = 20*1e-3; % [m]

% variable 5
% FinishType_min = .2;
% FinishType_max = 1;

% variable 6
% DebrisRemoved_min = 1;
% DebrisRemoved_max = 22;



tic

%% Algorithm parameters
numDesignVars = 3;
deltaX = 0.0001;
NG=5; %num generations
mrate=0.0001;   %mutation rates
N=200;   % population size
%%

BITS_hall_thruster=ceil(log((HallThruster_max-HallThruster_min)/deltaX)/log(2));
BITS_ion_thruster=ceil(log((IonThruster_max-IonThruster_min)/deltaX)/log(2));
% BITS_insul_thk = ceil(log((Insul_thk_max-Insul_thk_min)/deltaX)/log(2));
% BITS_finish = ceil(log((FinishType_max-FinishType_min)/deltaX)/log(2));
% BITS_debrisremoved = ceil(log((DebrisRemoved_max-DebrisRemoved_min)/deltaX)/log(2));
BITS = max([BITS_ion_thruster,BITS_hall_thruster]);

Xp = zeros(N,numDesignVars,1);
Xp(:,1,1)=ceil(rand(N,1)+0.5); %thruster type
Xp(:,2,1)=(HallThruster_max-HallThruster_min)*(rand(N,1))+HallThruster_min;
Xp(:,3,1)=ceil(rand(N,1)*3); %propellant selection
% Xp(:,4,1)=(Insul_thk_max-Insul_thk_min)*(rand(N,1))+Insul_thk_min;
% Xp(:,5,1)=FinishType_max*(rand(N,1))+FinishType_min;
% Xp(:,6,1)=DebrisRemoved_max*(rand(N,1))+DebrisRemoved_min;

XpOriginal = Xp;


% encode initial population

for ind = 1: N
    X1g = GA_encode(Xp(ind,1,:),1,2,BITS); 
    X2g = GA_encode(Xp(ind,2,:),HallThruster_min,HallThruster_max,BITS);
    X3g = GA_encode(Xp(ind,3,:),1,3,BITS);
    % X4g = GA_encode(Xp(ind,4,:),Insul_thk_min,Insul_thk_max,BITS);
    % X5g = GA_encode(Xp(ind,5,:),FinishType_min,FinishType_max,BITS);
    % X6g = GA_encode(Xp(ind,6,:),DebrisRemoved_min,DebrisRemoved_max,BITS);
    Xg(ind,:,1)=[X1g X2g X3g];
end

% evaluate initial population

for ind = 1: N
    F(ind,1)= -MDOrun(Xp(ind,1,1),Xp(ind,2,1),Xp(ind,3,1)); %only grabs cost, despite multiple outputs of MDOrun
end
% Favg(1)=mean(F(:,1));

% plot initial population


% [Xs,Ys,Zs] = meshgrid(-5:deltaX*5:5,-5:deltaX*5:5,-5:deltaX*5:5);
% F_brute_force = Q3GA(Xs,Ys,Zs); 
% [minX,indX] = min(F_brute_force);
% [minY,indY] = min(minX);
% [minZ,indZ] = min(minY);
%[indX,indY,indZ]

%disp('Look at initial population - and type return to start')

%keyboard

gen=1;
while gen<NG
    % selection - use Roulette Wheel Selection scheme
    
    R=max(F(:,gen))-min(F(:,gen));
    for ind=1:N
        FS(ind)=(F(ind,gen)-min(F(:,gen)))/R;
        FS(ind)=FS(ind).^2;
    end
    
    disp('Debug 1')
   
    Xkeep=[]; ind=1;
    while size(Xkeep,1)<N

        size(Xkeep,1)
        ind
        trial=rand;
        if FS(ind)>trial
            Xkeep=[Xkeep; Xg(ind,:,gen)];
        end
    
        ind=ind+1;
        ind=mod(ind,N);
        if ind==0
            ind=1;
        end


    end
    disp('Debug 4')
    
    % crossover - use provided crossover operator
    
    [Xgn]=GA_xover(Xkeep,0.95);
    
    % mutate
    
    [Xgnm]=GA_mutate(Xgn,mrate);
    
    % decode new population
    
    for ind=1:N
        Xpn1=round(GA_decode(Xgnm(ind,1:BITS),1,2,BITS));
        Xpn2=GA_decode(Xgnm(ind,BITS+1:2*BITS),HallThruster_min,HallThruster_max,BITS);
        Xpn3=round(GA_decode(Xgnm(ind,1:BITS),1,3,BITS));
        % Xpn4=GA_decode(Xgnm(ind,2*BITS+1:3*BITS),Insul_thk_min,Insul_thk_max,BITS);
        % Xpn5=GA_decode(Xgnm(ind,3*BITS+1:4*BITS),FinishType_min,FinishType_max,BITS);
        % Xpn6=GA_decode(Xgnm(ind,4*BITS+1:5*BITS),DebrisRemoved_min,DebrisRemoved_max,BITS);
        Xpn(ind,:)=[Xpn1 Xpn2 Xpn3];
    end


    % insert new population
    
    gen=gen+1;
    
    Xg(:,:,gen)=Xgnm;

    Xp(:,:,gen)=Xpn;
    
    % Compute Fitness of new population
    
    for ind = 1:N
        thrusterCurr = Xp(ind,1,gen);
        powerCurr = Xp(ind,2,gen);
        
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

        % debrisRemovedCurr = Xp(ind,6,gen)
        % if debrisRemovedCurr < 0;
        %     debrisRemovedCurr = DebrisRemoved_min;
        % end
        propellantCurr = Xp(ind, 3,gen);
        % insulCurr = Xp(ind,4,gen);
        % finishCurr = Xp(ind,5,gen);
        F(ind,gen) = -MDOrun(thrusterCurr,powerCurr,propellantCurr);
    end
  
    % Favg(gen)=mean(F(:,gen));
    
    
    
end
%% plotting 
% Favg = abs(Favg);
maxFitnesses = zeros(1,NG);
minFitnesses = zeros(1,NG);
for i = 1:NG
    maxFitnesses(i) = max(F(:,i));
    minFitnesses(i) = min(F(:,i));
end
disp('Finished GA')
toc

set(groot,'defaultAxesTickLabelInterpreter','latex');

set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');


figure(1)
clf(1)
%plot(Favg,LineWidth=1.5)
hold on
plot(-maxFitnesses/1e6,LineWidth=3, Color="#80B3FF")
%plot(-minFitnesses,LineWidth=1.5)
plot(NG,-maxFitnesses(NG)/1e6,'ok',LineWidth=4)
set(gcf, 'color','w')
box on
grid on
ax = gca;
ax.FontSize = 18;
xlabel('\textbf{Generation}', 'Interpreter','latex', 'FontSize',22)
ylabel('\textbf{Cost (\$USD million)}', 'Interpreter','latex', 'FontSize',22)
title('\textbf{Life Cycle Cost over Generations of Genetic Algorithm}', 'Interpreter','Latex', 'FontSize', 26)
legend('Minimum Cost',...
   sprintf('Minimum Cost of Final Generation = %.0f million',-maxFitnesses(NG)/1e6), 'Interpreter', 'latex', 'FontSize', 24);

%%  'Function Maximum Value',... 'Function Average Value',
[bestFitness,indBestFitness] = min(-F(:,NG));

BestProps = Xp(indBestFitness,:,NG)
disp('Optimal Thruster type (1 = Hall, 2 = Gridded Ion)')
disp(BestProps(1))
disp('Optimal Power to Propulsion in kW')
disp(BestProps(2)/1000)
disp('Optimal propellant type (1 = xenon, 2 = krypton, 3 = argon')
disp(BestProps(3))
% disp('Optimal insulation thickness')
% disp(BestProps(4))
% disp('Optimal surface absorbitivy')
% disp(BestProps(5))
% disp('Optimal number of debris removed per spacecraft')
% disp(BestProps(6))

disp('Before the thing')
[bestCost,bestTime,bestMass, bestMassFractions, bestMassFractionsSubsys, bestCostFractions, bestCostFractionsSubsys, bestCostFractionsSC, bestPropFtdrag]...
    = MDOrun(BestProps(1),BestProps(2),BestProps(3))
disp('After the thing')

%% Pie charts

% Specify custom colors
customColors = [
    0.8500 0.3250 0.0980; % Dark Red
    0.9290 0.6940 0.1250; % Orange/Yellow
    0.6350 0.0780 0.1840; % Deep Red
    0.3010 0.7450 0.9330; % Light Blue
    0.4660 0.6740 0.1880; % Dark Green
    0 0 1;               % Blue (for Slice 6)
    0 0.5 1;             % Light Blue (for Slice 7)
    0 0.25 0.5;          % Dark Blue (for Slice 8)
];


figure(2)
labels = {'Power System', 'Collection System', 'Propulsion System', 'Propellant'};
pie(bestMassFractions, labels)

figure(3)
figure('Position', [100, 100, 800, 600]); % Adjust the position and size as needed
labels2 = {'Battery', 'Solar Arrays', 'Propellant', 'Thrusters', 'PPU'};
pie(bestMassFractionsSubsys, labels2)
hText = findobj(gca, 'Type', 'text');  % Get handles to text objects
set(hText, 'Visible', 'on');  % Show all text labels
colormap(customColors)


%% excel export

if excelflag == 1

timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
folderName = ['OPTga_results/ignore_', timestamp];
mkdir(folderName);

filename = [folderName, '/telemetry.csv'];

% Define optimization parameters
numGenerations = NG;
mutationRate = mrate;
populationSize = N;
deltaXValue = deltaX;

% Create a cell array to hold the data
dataCell = cell(NG + 2, 8);  % Specify the number of rows and columns

% Add the optimization parameters
dataCell{1, 1} = 'Optimization Parameters';
dataCell{2, 1} = '# of Generations';
dataCell{2, 2} = 'Mutation Rate';
dataCell{2, 3} = 'Population Size';
dataCell{2, 4} = 'DeltaX';
dataCell{3, 1} = num2str(numGenerations);
dataCell{3, 2} = mrate;
dataCell{3, 3} = num2str(populationSize);
dataCell{3, 4} = deltaXValue;

% Insert an empty row
dataCell{4, 1} = '';  % Empty cell

% Add in spacecraft mass, time and cost 
dataCell{5, 1} = 'Optimal Outputs';
dataCell{6, 1} = 'Cost ($)';
dataCell{6, 2} = 'Time (years)';
dataCell{6, 3} = 'Spacecraft Mass (kg)';

dataCell{7, 1} = bestCost;
dataCell{7, 2} = bestTime;
dataCell{7, 3} = bestMass;

% Insert an empty row
dataCell{8, 1} = '';  % Empty cell

% Add in optimal design vector
dataCell{9, 1} = 'Optimal Design Vector';
dataCell{10, 1} = 'Thruster Type (1 = Hall Thruster 2 = Ion Thruster)';
dataCell{10, 2} = 'Optimal Power to Propulsion (W)';
dataCell{10, 3} = 'Propellant Type (1 = Xenon 2 = Krypton 3 = Argon)';
dataCell{10, 4} = 'Insulation Thickness';
dataCell{10, 5} = 'Surface Absorbtivity';
dataCell{10, 6} = 'Optimal number of debris removed per spacecraft';

dataCell{11, 1} = num2str(BestProps(1));
dataCell{11, 2} = num2str(BestProps(2));
dataCell{11, 3} = num2str(BestProps(3));
dataCell{11, 4} = num2str(BestProps(4));
dataCell{11, 5} = num2str(BestProps(5));
dataCell{11, 6} = num2str(BestProps(6));

% Insert an empty row
dataCell{12, 1} = '';  % Empty cell

% Add in mass fractions
dataCell{13, 1} = 'Mass Fraction (kg)';
dataCell{14,1} = 'Power System';
dataCell{14,2} = 'Thermal System';
dataCell{14,3} = 'Collection System';
dataCell{14,4} = 'Propulsion System';
dataCell{14,5} = 'Propellant';

dataCell{15,1} = bestMassFractions(1);
dataCell{15,2} = bestMassFractions(2);
dataCell{15,3} = num2str(bestMassFractions(3));
dataCell{15,4} = bestMassFractions(4);
dataCell{15,5} = bestMassFractions(5);

% Insert an empty row
dataCell{16, 1} = '';  % Empty cell

%Add in subsystem mass fractions
dataCell{17, 1} = 'Mass Fractions of subsystems (kg)';
dataCell{18,1} = 'Thermal Subsystem';
dataCell{18,2} = 'Battery';
dataCell{18,3} = 'Solar Panels';
dataCell{18,4} = 'Propellant';
dataCell{18,5} = 'Thrusters';
dataCell{18,6} = 'PPU';

dataCell{19,1} = bestMassFractions(2);       %thermal
dataCell{19,2} = bestMassFractionsSubsys(4); %battery
dataCell{19,3} = bestMassFractionsSubsys(5); %solar panel
dataCell{19,4} = bestMassFractionsSubsys(6); %propellant
dataCell{19,5} = bestMassFractionsSubsys(7); %thrusters
dataCell{19,6} = bestMassFractionsSubsys(8); %PPU

% Insert an empty row
dataCell{20, 1} = '';  % Empty cell

% Add info for cost fractions
dataCell{21, 1} = 'Cost Fractions (High Level, gives Total cost)';
dataCell{22, 1} = 'Spacecraft Costs ($) (individual sc * num spacecraft needed)';
dataCell{22, 2} = 'Upfront Costs ($)';
dataCell{22, 3} = 'Total Operational Costs ($)';
dataCell{22, 4} = 'DDTE Costs ($)';
bestCostFractions(1)
dataCell{23, 1} = bestCostFractions(1);
dataCell{23, 2} = bestCostFractions(2);
dataCell{23, 3} = bestCostFractions(3);
dataCell{23, 4} = bestCostFractions(4);

%Insert an empty row
dataCell{24, 1} = '';  % Empty cell

% Add info for spacecraft cost fractions
dataCell{25, 1} = 'Cost Fractions (Spacecraft Level Subsystems)';
dataCell{26, 1} = 'Collection ($)';
dataCell{26, 2} = 'Thruster ($)';
dataCell{26, 3} = 'Propellant ($)';
dataCell{26, 4} = 'Power System ($)';
dataCell{26, 5} = 'Structures & Thermal ($)';
dataCell{26, 6} = 'ADCS ($)';


dataCell{27, 1} = bestCostFractionsSubsys(1);
dataCell{27, 2} = bestCostFractionsSubsys(2);
dataCell{27, 3} = bestCostFractionsSubsys(3);
dataCell{27, 4} = bestCostFractionsSubsys(4);
dataCell{27, 5} = bestCostFractionsSubsys(5);
dataCell{27, 6} = num2str(bestCostFractionsSubsys(6));

%Insert an empty row
dataCell{28, 1} = '';  % Empty cell

dataCell{29, 1} = 'Cost of individual spacecrafts';
dataCell{30, 1} = 'Launch Costs ($)';
dataCell{30, 2} = 'Hardware Costs ($)';
dataCell{30, 3} = 'Propellant Costs ($)';

dataCell{31, 1} = bestCostFractionsSC(1);
dataCell{31, 2} = bestCostFractionsSC(2);
dataCell{31, 3} = bestCostFractionsSC(3);

%Insert an empty row
dataCell{32, 1} = '';  % Empty cell

dataCell{33, 1} = 'Propulsion Drag/thrust force';
dataCell{34, 1} = 'Thrust (N)';
dataCell{34, 2} = 'Drag (N)';

dataCell{35, 1} = bestPropFtdrag(1);
dataCell{35, 2} = bestPropFtdrag(2);

%Insert an empty row
dataCell{36, 1} = '';  % Empty cell

% Add the titles for the data
dataCell{37, 1} = 'Generation';
dataCell{37, 2} = 'minCost ($)';
dataCell{37, 3} = 'Date';

% Add the data
for i = 1:NG
    dataCell{i + 37, 1} = num2str(i);
    if i == 1
        dataCell{i + 37, 3} = datestr(datetime('now'));
    else 
        dataCell{i+37, 3} = '';
    end
    dataCell{i + 37, 2} = -maxFitnesses(i);
end

% Write the cell array to the CSV file
%cell2csv(filename, dataCell, ',');

% Save plots in the new folder
saveas(figure(1), fullfile(folderName, 'GA.png'));
saveas(figure(2), fullfile(folderName, 'massfractions.png'));

% Display the folder name
disp(['Results saved in folder: ' folderName]);

end

beep






