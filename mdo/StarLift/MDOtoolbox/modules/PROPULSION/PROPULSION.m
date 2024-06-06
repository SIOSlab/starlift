%% Giuliana
% Last edited by Max Luo, 4/9/24

function propulsion_out = PROPULSION(design_variables, power_out, parameter)

%% parameters
g           = parameter.g;                  %acceleration due to gravity,              [9.81 m/s^2]
mpay        = parameter.mpay;               %mass of payload                           [kg]

excelPath = "G:\My Drive\College\Grad school\ME Aerospace\Project\StarliftMDO\Starlift-MDO\Libraries";
addpath(excelPath);
excelName = "PARAMETERS_LIBRARY.xlsx";
sheetNames = sheetnames(excelName);

T = readtable(excelName, 'Sheet', sheetNames{2});
S = table2struct(T);
EPprops = {}; % records the names of propellants
EPmolarmass = []; % records the actual molar mass of each propellant

for i = 1:length(S)
    EPprops{i} = S(i).PROPELLANT;
    EPmolarmass(i) = S(i).MOLARMASS;
end
%% THIS NEEDS TO BE CHANGED VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
orbits_out = ORBITS(400000, 35786000 ,parameter); % LEO to GEO transfer
dV = orbits_out.dVConstThrust;
% ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cd          = parameter.Cd;                 %coefficient of drag                       [ - ]
r_bus       = parameter.r_bus;              %radius of the spacecraft bus              [m]

%% Hard coded, needs to be replaced
v = 7672.4; %velocity at old debris radius
rho = 2.03e-12; %Density of air at 500 km altitude. HARD CODED, NEEDS TO BE REPLACED

%% inputs
% dV_Transit  = traj_Out.dV_transit;         %DeltaV from debris to disposal orbit,     [m/s]
% dV_Launch   = traj_Out.dV_launch;          %DeltaV from initial orbit to debris orbit,[m/s]
mpowersys   = power_out.mass_powersys;     %mass of power system,   [kg]

%mass fractions
massfractionsPower      = power_out.massfractionsPower;

%% design variables
powerInput     = design_variables.disc_power;     %Power to Propulsion,                    [W] 
thrusterType   = design_variables.thruster_type;  %[1] Hall Thruster and [2] Gridded Ion,  [-]
propellant     = design_variables.propellant;     %[1] Xenon [2] Krypton [3] Iodine,       [-]
 

%% Calculate Thrust and Isp given empirical relationships of T/P and Isp/P
thrust = 0;
%calculate thrust/Isp with given power input

%% THIS WHOLE SECTION NEEDS TO BE MADE MORE FLEXIBLE%%%%%%%%%%%%%%%%%%%%%%%
switch thrusterType
    case 1 %hall thruster
                 switch propellant
                     case 1 %xenon
                        thrust =  0.001 * (.0521*powerInput + 6.2041); %review of alt propellants paper
                        Isp    =  279.9*log(powerInput) - 222.87;
                        %disp('xenon used');
                     case 2 %krypton
                         thrust = 0.001 * (0.0336*powerInput + 15.662); %review of propellants paper
                         Isp    =  479.5*log(powerInput) - 1569.8;
                         %disp('krypton used');
                     case 3 %argon
                         thrust = 0.001 * (.0051*powerInput + 98.5426);
                         Isp    =  326.65*log(powerInput) - 1695.5;
                         %disp('iodine used');
                 end
                 if Isp < 0
                     disp('bad bad')
                 end
    case 2 %DC ion thruster
        [thrust, Isp] = DCIonThruster(powerInput,EPmolarmass(propellant));
                
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate dry mass of propulsion system from thrust
   %this eq is based on empirical data
   %thrust in N, propulsion system dry mass in kg


% mass of singular thruster * number of thrusters
mThruster = 50.694*thrust + .4135;


mPPU = 1.74*(powerInput/1e3)+4.65; %https://arc-aiaa-org.proxy.library.cornell.edu/doi/pdf/10.2514/1.B34525

mdryPS = mThruster+mPPU;

%add propulsion system dry mass to spacecraft dry mass to get total mdry

mdry = mdryPS + mpowersys + mpay;

mp = mdry * (exp(dV/(Isp * g)) - 1);



%  mpTotal
%  mdry
%  Isp
%  debrisRemoved
%  deltaVtotal

%calculate wet mass 
mwet = mdry + mp;


%% drag constraint pt 1
% if mwet > 5.5e3
%     mp = 1e44;
% end

%% calculate drag constraint pt 2
A = pi * r_bus^2;
Fd = .5 * rho * Cd * A*v^2;

% if thrust < 2*Fd
%     mp = 1e45;
% end

%% Calculate time of mission, implement constraint
missionTime = (mp*Isp*g)/thrust;
% 
% if missionTime > 1.389e8 %4.4 years
%     mp = 1e44;
% end

%% mass outputs

massfractionsProp = [mp, mThruster, mPPU];
massfractions = [mpowersys, mpay, mdryPS, mp];
massfractionsSubsystem = [massfractionsPower, massfractionsProp];

%% outputs
propulsion_out.m_dry               = mdry;
propulsion_out.mission_time        = missionTime;
propulsion_out.m_wet               = mwet;
propulsion_out.m_propellant        = mp;
propulsion_out.massfractions       = massfractions;
propulsion_out.massfractionsSubsys = massfractionsSubsystem;
propulsion_out.Ftdrag              = [thrust, Fd];
end