%% Max

function cost_out = COST(design_variables,propulsion_out, power_out, parameter)

mass_wet = propulsion_out.m_wet; %[kg]
mass_dry = propulsion_out.m_dry; %[kg]
mass_propellant  = propulsion_out.m_propellant; % [kg] 
battery_capacity = power_out.battery_capacity; %[Wh]
cost_collection = parameter.payloadCost; % [$] 
thruster_power = design_variables.disc_power; % [W]
% total_debris = parameter.total_debris; % integer
launchWeight = parameter.launchWeight; %[kg]
launchCost = parameter.launchCost; %[$]                 %% INCLUDE ABILITY TO CHOOSE DIFFERENT LAUNCH PROVIDERS
% FalconHeavytoSSO = parameter.FalconHeavytoSSO; %[kg]
% FalconHeavyCost = parameter.FalconHeavyCost; %[$]
mission_time = propulsion_out.mission_time/60/60/24/365.25; %[y]
ops_cost = parameter.ops_cost; % [$/yr]
up_front_cost = parameter.up_front_cost; % [$]

% propellant = design_variables.propellant; % HARD CODED NEEDS TO BE CHANGED
propellant = 1;


uplink_time = 4; %hours. the amount of time needed to connect to the spacecraft to execute commands. used with AWS uplink services. built-in factor of safety
uplink_cost = (10 + 32)/2; % [$/min]. AWS ground station pricing. averaged between the two on-demand prices.

Q = parameter.Q;
S = parameter.S;
IOC = parameter.IOC;
B = parameter.B;
D = parameter.D;




thrustPower = design_variables.disc_power;

salary = 110000; % [$/yr] picked representative value

%% New models. make sure to scale all SMAD models by inflation
infl = 1.42; %scaling for inflation

%% Propulsion cost
switch propellant %from ken/michigan/max's paper
    case 1 %xenon
        cost_prop = (mass_propellant * 3000); %cited from michigan presentation
        %cost_prop = (mass_propellant* 14008); %from Ken Unfried, EFC
    case 2 %kryton
        cost_prop = (mass_propellant * 573); %from Ken Unfried, EFC
    case 3 %argon
        cost_prop = (mass_propellant * 29.55); %from Ken Unfried, EFC
end


cost_thruster = 1e6*(10.4*log(thruster_power/1000)-1.8);
                        % Cost of 2 string thruster based on approximate
                        % data reduction from fig 6 of this paper:
                        % https://arc.aiaa.org/doi/pdf/10.2514/1.B34525
                        % converts power in kW to cost in $


%% New power
% SMAD page 298. New SMAD was published in ~2011, so models are about 12
% years out of date
% Using Redwire product as model https://redwirespace.com/wp-content/uploads/2023/06/redwire-rigid-panel-solar-arrays-flysheet.pdf

%weight in kg (so....mass?)
solar_weight =  thrustPower * 0.15; % solar panel, ranges from 0.02 kg/W to 0.15 kg/W, as per redwire. Using 0.15 kg/W as conservative case
battery_weight = battery_capacity / 110; % capacity in [Wh]. approx 110 Wh/kg, estimate from https://www.mdpi.com/1996-1073/13/16/4097
harness_weight = 0; % this article says harness weight for small spacecraft can range from 4 to 15% of dry mass: https://arc.aiaa.org/doi/pdf/10.2514/1.48078?casa_token=Zdyq-oyFTWgAAAAA:cG83OoHs1JkvbTvblVismXZw0Ikep-4XjjEvObBivnX_wQbhOppyhmELLEEn254kLU5FaMjl8w
power_weight = solar_weight + battery_weight + harness_weight;

power_system_cost = infl * (64.3 * power_weight); % [$] sum total power system cost, SMAD model.

% old power
% cost_panel = area_panel *(400/((4/100)*(8/100))); % [$] $400 for 4x8cm^2 solar panel aray (NASA) https://spinoff.nasa.gov/Spinoff2016/ee_5.html

%% New weight and thermal

%dry mass includes thruster, thermal, PPU, collector

structure_thermal_cost = infl * (646 * (mass_dry)^.684); %SMAD models structure and thermal together
ADCS_cost = 15000; %https://www.cubesatshop.com/product-category/attitude-sensors/ estimate
% SMAD says 324 * ADCS_weight; no ADCS for now/make an assumption, but not
% a design variable

%% New harware total
cost_hardware = cost_collection + cost_thruster + power_system_cost + structure_thermal_cost + ADCS_cost; % $[m] sum total hardware costs

%% Ops cost
cost_ops = salary*2 + uplink_cost*(uplink_time*60); %debris mission previously derived from chapters 14 and 18 of https://onlinelibrary.wiley.com/doi/book/10.1002/9781119971009


%% launch cost
if mass_wet < launchWeight
    cost_launch = mass_wet  * launchCost/launchWeight;  %SpaceX: $67 million / 11000 kg to SSO
    %disp('falcon9')
% elseif mass_wet < FalconHeavytoSSO
%     cost_launch = mass_wet * FalconHeavyCost/FalconHeavytoSSO;
    %disp('falconheavy')
else
    cost_launch = -1;
end


cost_individual_spacecraft = cost_launch + cost_hardware + cost_prop; % $[m] sum total costs

%% Design, Development, Test, and Evaluation (DDT&E) cost estimation

DDTE = infl*1e6*(9.51*10^-4*1^(.59)*mass_dry^(0.66)*(80.6^S)*(3.81*10^-55)^(1/(IOC -1900))*B^(-.36)*1.57^D);

operations_cost = ops_cost*mission_time + cost_ops;

spacecraft_costs = cost_individual_spacecraft;

%% output 
total_cost                    = cost_individual_spacecraft + operations_cost+DDTE;

cost_fractions                = [spacecraft_costs, up_front_cost,operations_cost, DDTE];
cost_fractionsSubsys          = [cost_collection, cost_thruster,cost_prop, power_system_cost, structure_thermal_cost, ADCS_cost];
cost_spacecraft               = [cost_launch, cost_hardware , cost_prop];


cost_out.total_cost                    = total_cost;
cost_out.cost_fractions                = cost_fractions;
cost_out.cost_fractionsSubsys          = cost_fractionsSubsys;
cost_out.cost_spacecraft               = cost_spacecraft;




end
