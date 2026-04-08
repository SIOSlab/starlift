% Arthur Chadwick
% StarLift Multidisciplinary Design Optimization Iteration 2
% 6/7/2024

%% Input Choices:
    % Launch Vehicle can be Falcon Heavy, Starship, or New Glenn.
    % Resource Hub Orbit can be Low Earth Orbit or High Earth Orbit.
    % Number of Resource Hubs can be a discrete number being 1 to 100.
    % Propulsion System can be Chemical or Electric.
    % Starting Orbit can be Low Earth Orbit, High Earth Orbit, or Planetary.

%% Outputs:
    % Trajectory Architecture is outputted as Hohmann Transfer or Continuous Low-Thrust.
    % Resource Hub Mass of Each Subsystem.
    % Mass of Each Subsystem.
    % Power of Each Subsystem.
    % Cost of Total Mission.

%% MDO Code:
clearvars
close all
clc

for constraints = 1:1
    starting_orbit = "High Earth Orbit";
end

for design_choice = 1:1
    resource_hub_launch_vehicle = "Starship";
    resource_hub_orbit = "High Earth Orbit";
    number_of_resource_hubs = 10;
    resource_hub_propulsion_system = "Chemical";
    resource_hub_propulsion_Isp = 4;
    launch_vehicle = "Falcon Heavy";
    propulsion_system = "Electric";
    propulsion_Isp = 1;
    propulsion_power_level = 1;
    propulsion_ion_engine_type = 1;
    propulsion_hall_thruster_type = 1;
end

for shared_parameters = 1:1
    n = 100; %total # trajectory steps
    deltav = 4300./n; %[m/s] Total Change in Velocity the spacecraft should do (from elaine's class lecture for LEO-GEO)
    g = 9.81; %[m/s^2] Initial Gravitational Acceleration
    rE = 6378100; %[m] Radius of Earth
    G = 6.67408.*10.^-11; %[m^3/(kg*s^2)] Gravitational Constant
    mE = 5.972.*10.^24; %[kg] Mass of Earth
    cD = 0.1; %[-] spacecraft coefficient of drag
    A = 1; %[m^2] Frontal area of spacecraft
    initial_time = 0; %[s] initial time of simulation
    final_time_national_security = 60.*60.*24.*7; %[s] final time of simulation (1 week) National Security (CONSTRAINT)
    Pe = 1.*10.^-2; %[Pa] spaceraft thruster plume's pressure
    Ae = 1; %[m^2] spacecraft thruster nozzle's exit area
end

for resource_hub_launch_module = 1:1
        %Launch Vehicle Type
        if resource_hub_launch_vehicle == "Falcon Heavy" %SpaceX Falcon Heavy
            resource_hub_max_payload_to_leo = 63800; %[kg] spacex.com/vehicles/falcon-heavy
            resource_hub_max_payload_to_gto = 26700; %[kg] spacex.com/vehicles/falcon-heavy
            resource_hub_max_cylindrical_fairing_height = 13.1; %[m] spacex.com/vehicles/falcon-heavy
            resource_hub_max_cylindrical_fairing_diameter = 5.2; %[m] spacex.com/vehicles/falcon-heavy
            resource_hub_launch_cost = 90000000; %[$] nstxl.org/reducing-the-cost-of-space-travel-with-reusable-launch-vehicles/#:~:text=February%2012%2C%202024&text=SpaceX's%20Falcon%209%20rocket%20launches,of%20%2490%20million%20per%20launch.
        end
        if resource_hub_launch_vehicle == "Starship" %SpaceX Starship
            resource_hub_max_payload_to_leo = 150000; %[kg] wevolver.com/specs/spacexs-starship-sn24-bn7
            resource_hub_max_payload_to_gto = 21000; %[kg] wevolver.com/specs/spacexs-starship-sn24-bn7
            resource_hub_max_cylindrical_fairing_height = 18; %[m] wevolver.com/specs/spacexs-starship-sn24-bn7
            resource_hub_max_cylindrical_fairing_diameter = 9; %[m] wevolver.com/specs/spacexs-starship-sn24-bn7
            resource_hub_launch_cost = 100000000; %[$] payloadspace.com/payload-research-detailing-artemis-vehicle-rd-costs/
        end
        if resource_hub_launch_vehicle == "New Glenn" %Blue Origin New Glenn
            resource_hub_max_payload_to_leo = 45000; %[kg] blueorigin.com/new-glenn
            resource_hub_max_payload_to_gto = 13000; %[kg] blueorigin.com/new-glenn
            resource_hub_max_cylindrical_fairing_height = 6.383 + 4.1377 + 2.540; %[m] Figure 5-2: Standard capacity standard payload volume yellowdragonblog.com/wp-content/uploads/2019/01/new_glenn_payload_users_guide_rev_c.pdf
            resource_hub_max_cylindrical_fairing_diameter = 6.350; %[m] Figure 5-2: Standard capacity standard payload volume yellowdragonblog.com/wp-content/uploads/2019/01/new_glenn_payload_users_guide_rev_c.pdf
            resource_hub_launch_cost = 67000000; %[$] thespacereview.com/article/4626/1#:~:text=If%20New%20Glenn%20is%20priced,and%20becomes%20fully%20price%20competitive.
        end
        resource_hub_max_spacecraft_volume = pi*(resource_hub_max_cylindrical_fairing_diameter/2)*resource_hub_max_cylindrical_fairing_height; %[m^3]
        resource_hub_spacecraft_volume = resource_hub_max_spacecraft_volume; %[m^3]
end

for resource_hub_mass_module = 1:1
    if resource_hub_orbit == "Low Earth Orbit"
        resource_hub_initial_orbital_radius = 200000 + rE; %[m] LEO. https://www.nasa.gov/humans-in-space/leo-economy-frequently-asked-questions/#:~:text=Low%20Earth%20orbit%20(LEO)%20encompasses,communication%2C%20observation%2C%20and%20resupply.
        resource_hub_wet_mass = resource_hub_max_payload_to_leo;
        resource_hub_mass_scaling = 100/(100 + 27); %to get wet mass as 100%. The New SMAD, Table 14-18, page 422
        resource_hub_propellant_mass = resource_hub_wet_mass.*0.27.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_payload_mass = resource_hub_wet_mass.*0.31.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_structure_and_mechanisms_mass = resource_hub_wet_mass.*0.27.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_thermal_control_mass = resource_hub_wet_mass.*0.02.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_power_mass = resource_hub_wet_mass.*0.21.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_telemetry_tracking_and_command_mass = resource_hub_wet_mass.*0.02.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_on_board_processing_mass = resource_hub_wet_mass.*0.05.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_attitude_determination_and_control_mass = resource_hub_wet_mass.*0.06.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_propulsion_mass = resource_hub_wet_mass.*0.03.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_other_mass = resource_hub_wet_mass.*0.03.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
    end
    if resource_hub_orbit == "High Earth Orbit"
        resource_hub_initial_orbital_radius = 42164000; %[m] HEO. https://earthobservatory.nasa.gov/features/OrbitsCatalog/page2.php 
        resource_hub_wet_mass = resource_hub_max_payload_to_gto;
        resource_hub_mass_scaling = 100/(100 + 72); %to get wet mass as 100%. The New SMAD, Table 14-18, page 422
        resource_hub_propellant_mass = resource_hub_wet_mass.*0.72.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_payload_mass = resource_hub_wet_mass.*0.32.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_structure_and_mechanisms_mass = resource_hub_wet_mass.*0.24.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_thermal_control_mass = resource_hub_wet_mass.*0.04.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_power_mass = resource_hub_wet_mass.*0.17.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_telemetry_tracking_and_command_mass = resource_hub_wet_mass.*0.04.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_on_board_processing_mass = resource_hub_wet_mass.*0.03.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_attitude_determination_and_control_mass = resource_hub_wet_mass.*0.06.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_propulsion_mass = resource_hub_wet_mass.*0.07.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        resource_hub_other_mass = resource_hub_wet_mass.*0.03.*resource_hub_mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
    end
end

for resource_hub_propulsion_module = 1:1
    if resource_hub_propulsion_system == "Electric"
    %Propulsion Power
        if resource_hub_propulsion_power_level == 1 %Level 1
            resource_hub_propulsion_input_power_kw = 0.5; %[kW]
        end
        if resource_hub_propulsion_power_level == 2 %Level 2
            resource_hub_propulsion_input_power_kw = 0.25; %[kW]
        end
        if resource_hub_propulsion_power_level == 3 %Level 3
            resource_hub_propulsion_input_power_kw = 0.75; %[kW]
        end
    %Propulsion Type
        if resource_hub_propulsion_Isp == 1 %Electric Hall Thruster
            resource_hub_Isp = 1500; %[sec]
        end
        if resource_hub_propulsion_Isp == 2 %Electric Field Emission Thruster
            resource_hub_Isp = 1000; %[sec]
        end
        if resource_hub_propulsion_Isp == 3 %Electric Ion Engine Thruster
            resource_hub_Isp = 3000; %[sec]
        end
        %Ion Engine Type
            if resource_hub_propulsion_ion_engine_type == 1 %DC Ion Engine
                resource_hub_thrust = 33.98.*resource_hub_propulsion_input_power_kw; %[N] https://arc.aiaa.org/doi/full/10.2514/1.A33647?journalCode=jsr
            end
            if resource_hub_propulsion_ion_engine_type == 2 %RF Ion Engine
                resource_hub_thrust = 30.75.*resource_hub_propulsion_input_power_kw; %[N] https://arc.aiaa.org/doi/full/10.2514/1.A33647?journalCode=jsr
            end
        %Hall Thruster Type
            if resource_hub_propulsion_hall_thruster_type == 1 %SPT/TAL Hall
                resource_hub_thrust = 54.26.*resource_hub_propulsion_input_power_kw; %[N] https://arc.aiaa.org/doi/full/10.2514/1.A33647?journalCode=jsr
            end
            if resource_hub_propulsion_hall_thruster_type == 2 %Cylindrical Hall
                resource_hub_thrust = 33.47.*resource_hub_propulsion_input_power_kw; %[N] https://arc.aiaa.org/doi/full/10.2514/1.A33647?journalCode=jsr
            end

    end
    if resource_hub_propulsion_system == "Chemical"
        if resource_hub_propulsion_Isp == 4 %Chemical Monopropellant Thruster
            resource_hub_Isp = 200; %[sec]
        end
    end
end

for resource_hub_trajectory_module = 1:1
    if resource_hub_orbit == "Low Earth Orbit"

    end
    if resource_hub_orbit == "High Earth Orbit"
        
    end
end

for resource_hub_cost_module = 1:1
    resource_hub_launch_cost = number_of_resource_hubs.*resource_hub_launch_cost; %[$]
end

for launch_module = 1:1
        %Launch Vehicle Type
        if launch_vehicle == "Falcon Heavy" %SpaceX Falcon Heavy
            max_payload_to_leo = 63800; %[kg] spacex.com/vehicles/falcon-heavy
            max_payload_to_gto = 26700; %[kg] spacex.com/vehicles/falcon-heavy
            max_cylindrical_fairing_height = 13.1; %[m] spacex.com/vehicles/falcon-heavy
            max_cylindrical_fairing_diameter = 5.2; %[m] spacex.com/vehicles/falcon-heavy
            launch_cost = 90000000; %[$] nstxl.org/reducing-the-cost-of-space-travel-with-reusable-launch-vehicles/#:~:text=February%2012%2C%202024&text=SpaceX's%20Falcon%209%20rocket%20launches,of%20%2490%20million%20per%20launch.
        end
        if launch_vehicle == "Starship" %SpaceX Starship
            max_payload_to_leo = 150000; %[kg] wevolver.com/specs/spacexs-starship-sn24-bn7
            max_payload_to_gto = 21000; %[kg] wevolver.com/specs/spacexs-starship-sn24-bn7
            max_cylindrical_fairing_height = 18; %[m] wevolver.com/specs/spacexs-starship-sn24-bn7
            max_cylindrical_fairing_diameter = 9; %[m] wevolver.com/specs/spacexs-starship-sn24-bn7
            launch_cost = 100000000; %[$] payloadspace.com/payload-research-detailing-artemis-vehicle-rd-costs/
        end
        if launch_vehicle == "New Glenn" %Blue Origin New Glenn
            max_payload_to_leo = 45000; %[kg] blueorigin.com/new-glenn
            max_payload_to_gto = 13000; %[kg] blueorigin.com/new-glenn
            max_cylindrical_fairing_height = 6.383 + 4.1377 + 2.540; %[m] Figure 5-2: Standard capacity standard payload volume yellowdragonblog.com/wp-content/uploads/2019/01/new_glenn_payload_users_guide_rev_c.pdf
            max_cylindrical_fairing_diameter = 6.350; %[m] Figure 5-2: Standard capacity standard payload volume yellowdragonblog.com/wp-content/uploads/2019/01/new_glenn_payload_users_guide_rev_c.pdf
            launch_cost = 67000000; %[$] thespacereview.com/article/4626/1#:~:text=If%20New%20Glenn%20is%20priced,and%20becomes%20fully%20price%20competitive.
        end
        max_spacecraft_volume = pi*(max_cylindrical_fairing_diameter/2)*max_cylindrical_fairing_height; %[m^3]
        spacecraft_volume = max_spacecraft_volume; %[m^3]
end

for mass_module = 1:1
    if starting_orbit == "Low Earth Orbit"
        initial_orbital_radius = 200000 + rE; %[m] LEO. https://www.nasa.gov/humans-in-space/leo-economy-frequently-asked-questions/#:~:text=Low%20Earth%20orbit%20(LEO)%20encompasses,communication%2C%20observation%2C%20and%20resupply.
        spacecraft_wet_mass = max_payload_to_leo;
        mass_scaling = 100/(100 + 27); %to get wet mass as 100%. The New SMAD, Table 14-18, page 422
        propellant_mass = spacecraft_wet_mass.*0.27.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        payload_mass = spacecraft_wet_mass.*0.31.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        structure_and_mechanisms_mass = spacecraft_wet_mass.*0.27.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        thermal_control_mass = spacecraft_wet_mass.*0.02.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        power_mass = spacecraft_wet_mass.*0.21.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        telemetry_tracking_and_command_mass = spacecraft_wet_mass.*0.02.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        on_board_processing_mass = spacecraft_wet_mass.*0.05.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        attitude_determination_and_control_mass = spacecraft_wet_mass.*0.06.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        propulsion_mass = spacecraft_wet_mass.*0.03.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        other_mass = spacecraft_wet_mass.*0.03.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
    end
    if starting_orbit == "High Earth Orbit"
        initial_orbital_radius = 42164000; %[m] HEO. https://earthobservatory.nasa.gov/features/OrbitsCatalog/page2.php 
        spacecraft_wet_mass = max_payload_to_gto;
        mass_scaling = 100/(100 + 72); %to get wet mass as 100%. The New SMAD, Table 14-18, page 422
        propellant_mass = spacecraft_wet_mass.*0.72.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        payload_mass = spacecraft_wet_mass.*0.32.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        structure_and_mechanisms_mass = spacecraft_wet_mass.*0.24.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        thermal_control_mass = spacecraft_wet_mass.*0.04.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        power_mass = spacecraft_wet_mass.*0.17.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        telemetry_tracking_and_command_mass = spacecraft_wet_mass.*0.04.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        on_board_processing_mass = spacecraft_wet_mass.*0.03.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        attitude_determination_and_control_mass = spacecraft_wet_mass.*0.06.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        propulsion_mass = spacecraft_wet_mass.*0.07.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        other_mass = spacecraft_wet_mass.*0.03.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
    end
    if starting_orbit == "Planetary"
        spacecraft_wet_mass = max_payload_to_gto;
        mass_scaling = 100/(100 + 110); %to get wet mass as 100%. The New SMAD, Table 14-18, page 422
        propellant_mass = spacecraft_wet_mass.*1.10.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        payload_mass = spacecraft_wet_mass.*0.15.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        structure_and_mechanisms_mass = spacecraft_wet_mass.*0.25.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        thermal_control_mass = spacecraft_wet_mass.*0.06.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        power_mass = spacecraft_wet_mass.*0.21; %[kg] The New SMAD, Table 14-18, page 422
        telemetry_tracking_and_command_mass = spacecraft_wet_mass.*0.07.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        on_board_processing_mass = spacecraft_wet_mass.*0.04.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        attitude_determination_and_control_mass = spacecraft_wet_mass.*0.06.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        propulsion_mass = spacecraft_wet_mass.*0.13.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
        other_mass = spacecraft_wet_mass.*0.03.*mass_scaling; %[kg] The New SMAD, Table 14-18, page 422
    end
end

for power_module = 1:1
    if starting_orbit == "Low Earth Orbit"
        spacecraft_power = 794; %[W] The New SMAD, Table 14-20, page 424
        payload_power = spacecraft_power*0.46; %[W] The New SMAD, Table 14-20, page 424
        structure_and_mechanisms_power = spacecraft_power*0.01; %[W] The New SMAD, Table 14-20, page 424
        thermal_control_power = spacecraft_power*0.10; %[W] The New SMAD, Table 14-20, page 424
        power_power = spacecraft_power*0.09; %[W] The New SMAD, Table 14-20, page 424
        telemetry_tracking_and_command_power = spacecraft_power*0.12; %[W] The New SMAD, Table 14-20, page 424
        on_board_processing_power = spacecraft_power*0.12; %[W] The New SMAD, Table 14-20, page 424
        attitude_determination_and_control_power = spacecraft_power*0.10; %[W] The New SMAD, Table 14-20, page 424
        propulsion_power = spacecraft_power*0; %[W] The New SMAD, Table 14-20, page 424
    end
    if starting_orbit == "High Earth Orbit"
        spacecraft_power = 691; %[W] The New SMAD, Table 14-20, page 424
        payload_power = spacecraft_power*0.35; %[W] The New SMAD, Table 14-20, page 424
        structure_and_mechanisms_power = spacecraft_power*0; %[W] The New SMAD, Table 14-20, page 424
        thermal_control_power = spacecraft_power*0.14; %[W] The New SMAD, Table 14-20, page 424
        power_power = spacecraft_power*0.07; %[W] The New SMAD, Table 14-20, page 424
        telemetry_tracking_and_command_power = spacecraft_power*0.16; %[W] The New SMAD, Table 14-20, page 424
        on_board_processing_power = spacecraft_power*0.10; %[W] The New SMAD, Table 14-20, page 424
        attitude_determination_and_control_power = spacecraft_power*0.16; %[W] The New SMAD, Table 14-20, page 424
        propulsion_power = spacecraft_power*0.02; %[W] The New SMAD, Table 14-20, page 424
    end
    if starting_orbit == "Planetary"
        spacecraft_power = 749; %[W] The New SMAD, Table 14-20, page 424
        payload_power = spacecraft_power*0.22; %[W] The New SMAD, Table 14-20, page 424
        structure_and_mechanisms_power = spacecraft_power*0.01; %[W] The New SMAD, Table 14-20, page 424
        thermal_control_power = spacecraft_power*0.15; %[W] The New SMAD, Table 14-20, page 424
        power_power = spacecraft_power*0.10; %[W] The New SMAD, Table 14-20, page 424
        telemetry_tracking_and_command_power = spacecraft_power*0.18; %[W] The New SMAD, Table 14-20, page 424
        on_board_processing_power = spacecraft_power*0.11; %[W] The New SMAD, Table 14-20, page 424
        attitude_determination_and_control_power = spacecraft_power*0.12; %[W] The New SMAD, Table 14-20, page 424
        propulsion_power = spacecraft_power*0.11; %[W] The New SMAD, Table 14-20, page 424
    end
end

for propulsion_module = 1:1
    if propulsion_system == "Electric"
        trajectory_architecture = "Continuous Low-Thrust";
    %Propulsion Power
        if propulsion_power_level == 1 %Level 1
            propulsion_input_power_kw = 0.5; %[kW]
        end
        if propulsion_power_level == 2 %Level 2
            propulsion_input_power_kw = 0.25; %[kW]
        end
        if propulsion_power_level == 3 %Level 3
            propulsion_input_power_kw = 0.75; %[kW]
        end
    %Propulsion Type
        if propulsion_Isp == 1 %Electric Hall Thruster
            Isp = 1500; %[sec]
        end
        if propulsion_Isp == 2 %Electric Field Emission Thruster
            Isp = 1000; %[sec]
        end
        if propulsion_Isp == 3 %Electric Ion Engine Thruster
            Isp = 3000; %[sec]
        end
        %Ion Engine Type
            if propulsion_ion_engine_type == 1 %DC Ion Engine
                thrust = 33.98.*propulsion_input_power_kw; %[N] https://arc.aiaa.org/doi/full/10.2514/1.A33647?journalCode=jsr
            end
            if propulsion_ion_engine_type == 2 %RF Ion Engine
                thrust = 30.75.*propulsion_input_power_kw; %[N] https://arc.aiaa.org/doi/full/10.2514/1.A33647?journalCode=jsr
            end
        %Hall Thruster Type
            if propulsion_hall_thruster_type == 1 %SPT/TAL Hall
                thrust = 54.26.*propulsion_input_power_kw; %[N] https://arc.aiaa.org/doi/full/10.2514/1.A33647?journalCode=jsr
            end
            if propulsion_hall_thruster_type == 2 %Cylindrical Hall
                thrust = 33.47.*propulsion_input_power_kw; %[N] https://arc.aiaa.org/doi/full/10.2514/1.A33647?journalCode=jsr
            end
    end
    if propulsion_system == "Chemical"
        trajectory_architecture = "Hohmann Transfer";
        if propulsion_Isp == 4 %Chemical Monopropellant Thruster
            Isp = 200; %[sec]
        end
    end
end
       
for trajectory_module = 1:1
    %Incorporate Orbits states from Orbits team
end

for cost_module = 1:1
    cost = launch_cost; %[$]
    cost_including_resource_hub = resource_hub_launch_cost + cost; %[$]
end