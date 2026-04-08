function parameter=MDOgenerateParamaterSENS(step)

    % Trajectory 
    parameter.r_e=6.3781e6;% radius of Earth, m
    parameter.r_low=300e3+parameter.r_e; %Disposal Radius, m
    parameter.r_high=850e3+parameter.r_e; %Radius of Zenit-2 Upper Stage, m
    parameter.r_launch=185e3+parameter.r_e; %Launch Radius, m
    parameter.m_debris=8.5e3; %Mass of Zenit-2 Upper Stage, kg
    parameter.mue=3.986004418e14; %gravitational constant [m3 s−2]
    parameter.g = 9.81; %g, m/s^2
    parameter.Mxe = 131.3; %molecular mass of Xe, g/mol
    parameter.Mkr = 83.8; %molecular mass of Kr, g/mol
    parameter.Mar = 39.95; %molecular mass of Ar, g/mol

    % Power
    parameter.alpha = 1368; %solar incident radiation at Earth %W/m^2
    parameter.n = .29 + step(12); %solar panel efficiency,  0<x<1
    parameter.power_FoS = 1.25 + step(13); %power factor of safety
    parameter.batt_dens = 200 + step(14); %battery density Wh/kg; https://www.eaglepicher.com/markets/space/satellites/
    parameter.panel_density = 10 + step(15); %panel areal density kg/m^2 https://www.valispace.com/wp-content/uploads/2018/12/EPS-sizing-tutorial-1.pdf
    parameter.batt_min_charge = .3 + step(16); %minimum battery charge, 0<x<1;
    parameter.debris_body_diameter = 3.9; %[m]
    parameter.debris_body_length = 11.047; %[m], https://www.russianspaceweb.com/zenit_stage2.html
    parameter.A_debris= (parameter.debris_body_diameter/2)^2 * pi; %max surface area of zenit-2 upper stage, m^2 
    %parameter.A_debris= .5*parameter.debris_body_diameter*parameter.debris_body_length; %max surface area of zenit-2 upper stage, m^2
    parameter.rho_deorbitAlt = 1e-10 + step(20); %[kg/m^3], SMAD density of atmosphere at solar max, 200km alt
    parameter.Cd = 4 + step(21);  %Drag coefficient, worst case scenario SMAD
    parameter.velocity_deorbitAlt = 7672.4; %[m/s], velocity of body at 300km altitude

    
  
    parameter.spacecraft_density = 200 + step(23); %[kg/m^3] 
    % https://www.researchgate.net/figure/Lifetime-of-a-spherical-satellite-with-mass-density-02-g-cm-3-versus-altitude-and-size_fig1_231941864#:~:text=altitude%20and%20size%2C%20assuming%20average,%5BLoftus%20and%20Reynolds%201993%5D.
    
    % Collection 
    parameter.mass_collectsys = 71 + step(24); %[kg], via marcus and sedwick paper
   
    % Cost
    parameter.propellant_cost_per_kg = 3000 + step(25); % $xenon 3000/kg per https://cosmosmagazine.com/space/exploration/iodine-powered-spacecraft-tested-in-orbit/
    parameter.cost_collectsys = 5e6 + step(26);%[$] based on MEV architecture and assumptions about ratio of satellite cost to collection mechanism cost
    
    %Thermal Parameters:
    parameter.r_bus=0.5 + step(27); %Spacecraft Bus Radius

    parameter.alpha_I_list=[.2 .16 .17 .18 .22 .92 .34 .38 .41 .46];
    parameter.epsilon_I_list=[.85 .87 .92 .91 .88 .89 .84 .55 .67 .75 .86];
    %parameter.k=0.0000519; %W/m*K
    parameter.k=5e-5 + step(30); %W/m*K per https://iopscience.iop.org/article/10.1088/1757-899X/396/1/012061/pdf

    
    parameter.q_EarthIR_Inc_low=65 + step(31);
    parameter.q_EarthIR_Inc_high=50 + step(32);
    
    
    parameter.q_albedo_Inc_low_60=0.5*0.5*(350) + step(33); %When beta=60 deg, 0 when beta=90
    parameter.q_albedo_Inc_high_60=0.5*0.5*(100) + step(34); %When beta=60 deg, 0 when beta=90

    parameter.sigma=5.699e-8; %Stefan-Boltzman Constant, W/m^2 K^4
    
    
    
 
    
    parameter.q_solar_Inc_low_90=1414 + step(36); %When beta=90 deg
    parameter.q_solar_Inc_high_90=1414 + step(37); %When beta=90 deg
    
    parameter.alpha_R=.16 + step(38); %Absorptance of Finish, Diffuse Quartz
    parameter.epsilon_R=.80 + step(39); %Emitance of Finish, Diffuse Quartz

    parameter.T_E_cold=0+273 + step(40); %Minimum allowable electronics temperature, K
    parameter.T_E_hot=30+273 + step(41); %Maximum allowable electronics temperature, K

    parameter.total_debris = 22 + step(42);

    %% structures
    parameter.R_univ   = 8.314; %universal gas constant,               [J]
    parameter.To       = 300 + step(44);   %assumed tank temperature,             [K]
    parameter.rho_tank = 2700 + step(45);  %assumed tank density, aluminum alloy, [kg/m^3]
    parameter.yield    = 100e6 + step(46); %assumed tank yield strength,          [Pa]
    parameter.MWxe     = 131.3; %molecular weight of Xe                [g/mol]
    parameter.MWkr     = 83.8;  %molecular weight of Kr                [g/mol]
    parameter.MWar     = 39.95; %molecular weight of Ar                [g/mol]

    %% DDTE 
    parameter.Q       =2 + step(50);
    parameter.S      = 1.99 + step(51);
    parameter.IOC    = 2030 + step(52);
    parameter.B      = 3 + step(53);
    parameter.D      = 0 + step(54);


    %cost
    parameter.Falcon9toSSO = 11e3 + step(55);
    parameter.Falcon9Cost = 67e6 + step(56);

    parameter.FalconHeavytoSSO = 11e3*8/3.49 + step(57);
    parameter.FalconHeavyCost = 97e6 + step(58);

    parameter.ops_cost = 20.12e6 + step(59); %[$/yr]
    parameter.up_front_cost = 50e6 + step(60);

end