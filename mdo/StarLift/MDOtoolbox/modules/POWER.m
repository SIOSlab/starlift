%% ELLIOT
% Last edited by Max Luo, 3/20/24
function power_out = POWER(design_variables, parameter)

%% Parameters and Inputs
%parameters
alpha = parameter.alpha; %solar incident radiation at earth, [W/m^2]
panel_n = parameter.n; %solar panel efficiency, [0<x<1]
P_FoS = parameter.power_FoS; %power factor of safety, [-]
batt_dens = parameter.batt_dens; %battery density, [Wh/kg]
panel_density = parameter.panel_density; %panel areal density, [kg/m^2]
batt_min_charge = parameter.batt_min_charge; %minimum battery charge, [0<x<1]
orbit_period = parameter.T2; %orbital period max, [s]
Eclipse_time_ratio = parameter.eclipse_ratio_max; % max eclipse time fraction, [0<x<1]

%% Design variables and Power Requirements
% power requirements, assumed 100% duty cycle
prop_power_req = design_variables.disc_power; %[W] prop system takes up all of discretionary power budget during thrust maneuvers, which is max power case

other_power_req = .05 * prop_power_req; %[W] assume other spacecraft functionality (GNC, comms, etc.) requires 5% of prop power
P_reqavg = prop_power_req + other_power_req; %[W]


%% Calcuations
% power generation = average power * factor of / (1-fraction of time in
% eclipse) --> all power must be generated during light time
P_generation_req = P_reqavg*P_FoS/(1-Eclipse_time_ratio); %[W]

%panel area = required generation / (incident radiation*panel efficiency)
A_panel = P_generation_req/(alpha*panel_n); %[m^2]

% panel mass calculations
mass_panel = A_panel*panel_density; %[kg]

% battery must hold all power to be used over the course of an orbit, with
% a maximum discharge fraction 
% batt capactiy = P_req [W] * orbit period [s * min/s * hr/min = hr] /
% (1+battery min charge) = [Wh]
battery_capacity = P_generation_req*orbit_period/(60*60)*(1+batt_min_charge); % [Wh]
mass_batt = battery_capacity/batt_dens; %[kg]
mass_powersys = mass_batt + mass_panel; %[kg]


power_out.battery_capacity = battery_capacity; %[J]
power_out.power_generated = P_generation_req; %[W]
power_out.mass_powersys = mass_powersys; %[kg]
power_out.panel_area = A_panel; %[m^2]
power_out.massfractionsPower = [mass_batt, mass_panel];

end

