function [outVec]=SA_perturbCost(inVec)
% [E]=perturbAtoms(ri);
% Perturbs the atom configuration - simulated
% annealing sample problem

% index 1: thruster type
% index 2: thruster power
% index 3: insulation thk
% index 4: finish type .2 - 1
% index 5: debris removed 1 - 10

HallThruster_min = 500; %[W]
HallThruster_max = 3000; %[W]
IonThruster_min = 500; %[W]
IonThruster_max = 3000; %[W]
Insul_thk_min = 1/100; % [m]
Insul_thk_max = 15/100; % [m]
finish_max = 1;
finish_min = .2;

intChange = randi(5,1);

if intChange == 1
    inVec(1) = ceil(rand+.5);
end
if intChange == 2
    if inVec(1) ==1
        inVec(2) = HallThruster_min + rand*(HallThruster_max-HallThruster_min);
    else
        inVec(2) = IonThruster_min + rand*(IonThruster_max-IonThruster_min);
    end
end
if intChange == 3
    inVec(3) = Insul_thk_min + rand*(Insul_thk_max-Insul_thk_min);
end
if intChange == 4
   inVec(4) = finish_min + rand*(finish_max-finish_min);
end
if intChange == 5
   inVec(5) = 1+21*rand;
end

% move atom with index indp to slot with index inds
outVec=inVec;
