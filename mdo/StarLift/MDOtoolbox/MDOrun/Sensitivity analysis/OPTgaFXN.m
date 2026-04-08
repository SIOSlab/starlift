% Implement GA for Q3
% Adapted from
% dWo, 3/15/2002
% function
function [BestProps,bestCost,bestTime,bestMass, bestMassFractions, bestMassFractionsSubsys, bestCostFractions, bestCostFractionsSubsys, bestCostFractionsSC, bestPropFtdrag] = OPTgaFXN()


%bounds on continuous design variables/factors for genetic algorithm
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

deltaX = 0.0001;
NG=300; %num generations
BITS_hall_thruster=ceil(log((HallThruster_max-HallThruster_min)/deltaX)/log(2));
BITS_ion_thruster=ceil(log((IonThruster_max-IonThruster_min)/deltaX)/log(2));
BITS_insul_thk = ceil(log((Insul_thk_max-Insul_thk_min)/deltaX)/log(2));
BITS_finish = ceil(log((FinishType_max-FinishType_min)/deltaX)/log(2));
BITS_debrisremoved = ceil(log((DebrisRemoved_max-DebrisRemoved_min)/deltaX)/log(2));
BITS = max([BITS_insul_thk,BITS_ion_thruster,BITS_hall_thruster,BITS_finish]);
mrate=0.0001;   %mutation rates
N=200;   % population size



Xp = zeros(N,numDesignVars,1);
Xp(:,1,1)=ceil(rand(N,1)+0.5);
Xp(:,2,1)=(HallThruster_max-HallThruster_min)*(rand(N,1))+HallThruster_min;
Xp(:,3,1)=ceil(rand(N,1)*3);
Xp(:,4,1)=(Insul_thk_max-Insul_thk_min)*(rand(N,1))+Insul_thk_min;
Xp(:,5,1)=FinishType_max*(rand(N,1))+FinishType_min;
Xp(:,6,1)=DebrisRemoved_max*(rand(N,1))+DebrisRemoved_min;


% encode initial population

for ind = 1: N
    X1g = GA_encode(Xp(ind,1,:),1,2,BITS); 
    X2g = GA_encode(Xp(ind,2,:),HallThruster_min,HallThruster_max,BITS);
    X3g = GA_encode(Xp(ind,3,:),1,3,BITS);
    X4g = GA_encode(Xp(ind,4,:),Insul_thk_min,Insul_thk_max,BITS);
    X5g = GA_encode(Xp(ind,5,:),FinishType_min,FinishType_max,BITS);
    X6g = GA_encode(Xp(ind,6,:),DebrisRemoved_min,DebrisRemoved_max,BITS);
    Xg(ind,:,1)=[X1g X2g X3g X4g X5g X6g];
end

% evaluate initial population

for ind = 1: N
    F(ind,1)= -MDOrun(Xp(ind,1,1),Xp(ind,2,1),Xp(ind,3,1),Xp(ind,4,1),Xp(ind,5,1), Xp(ind,6,1));
end

Favg(1)=mean(F(:,1));
gen=1;

while gen<NG
    
    % selection - use Roulette Wheel Selection scheme
    
    R=max(F(:,gen))-min(F(:,gen));
    for ind=1:N
    FS(ind)=(F(ind,gen)-min(F(:,gen)))/R;
    FS(ind)=FS(ind).^2;
    end
    
   
    Xkeep=[]; ind=1;
    while size(Xkeep,1)<N
    trial=rand;
    if FS(ind)>trial
        Xkeep=[Xkeep ; Xg(ind,:,gen)];
    else
    sel=0;
    end
    
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
        elseif debrisRemovedCurr < 0
            debrisRemovedCurr = DebrisRemoved_min;
        end
        propellantCurr = Xp(ind, 3,gen);
        insulCurr = Xp(ind,4,gen);
        finishCurr = Xp(ind,5,gen);
        debrisRemovedCurr = Xp(ind,6,gen);
        F(ind,gen) = -MDOrun(thrusterCurr,powerCurr,propellantCurr,insulCurr,finishCurr,debrisRemovedCurr);
    end
  
    Favg(gen)=mean(F(:,gen));
end
%% plotting 
Favg = abs(Favg);
maxFitnesses = zeros(1,NG);
minFitnesses = zeros(1,NG);
for i = 1:NG
    maxFitnesses(i) = max(F(:,i));
    minFitnesses(i) = min(F(:,i));
end
disp('Finished GA\n')
toc

% set(groot,'defaultAxesTickLabelInterpreter','latex');
% 
% set(groot,'defaulttextinterpreter','latex');
% set(groot,'defaultLegendInterpreter','latex');

[bestFitness,indBestFitness] = min(-F(:,NG));

BestProps = Xp(indBestFitness,:,NG);
disp('Optimal Thruster type (1 = Hall, 2 = Gridded Ion)')
disp(BestProps(1))
disp('Optimal Power to Propulsion in kW')
disp(BestProps(2)/1000)
disp('Optimal propellant type (1 = xenon, 2 = krypton, 3 = argon')
disp(BestProps(3))
disp('Optimal insulation thickness')
disp(BestProps(4))
disp('Optimal surface absorbitivy')
disp(BestProps(5))
disp('Optimal number of debris removed per spacecraft')

[bestCost,bestTime,bestMass, bestMassFractions, bestMassFractionsSubsys, bestCostFractions, bestCostFractionsSubsys, bestCostFractionsSC, bestPropFtdrag]...
    = MDOrun(BestProps(1),BestProps(2),BestProps(3),BestProps(4),BestProps(5), BestProps(6));

end




