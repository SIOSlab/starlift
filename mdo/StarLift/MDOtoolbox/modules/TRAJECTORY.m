%% Alex / Elliot
function traj_Out=TRAJECTORY(parameter)

    % r_low=parameter.r_low; %Disposal Radius
    % r_high=parameter.r_high; %Radius of Zenit-2 Upper Stage
    % r_launch=parameter.r_launch; %Launch Radius
    mue=parameter.mue; %gravitational constant

    

    dV_transit=abs(sqrt(mue/r_low)-sqrt(mue/r_high));
    dV_launch=abs(sqrt(mue/r_launch)-sqrt(mue/r_high));
    
    T_min=2*pi*sqrt(r_launch^3/mue);
    T_max=2*pi*sqrt(r_high^3/mue);

    eclipse_ratio_min=0.25; %Eclipse Fraction of A Spacecraft in at 90 degree inclination with altitude of 850 km 
    eclipse_ratio_max=0.31; %Eclipse Fraction of A Spacecraft in at 90 degree inclination with altitude of 185 or 320 km 
    
    
    %Configure Outputs
    traj_Out.dV_transit=dV_transit;
    traj_Out.dV_launch=dV_launch;
    traj_Out.eclipse_ratio_min=eclipse_ratio_min;
    traj_Out.eclipse_ratio_max=eclipse_ratio_max;
    traj_Out.T_min=T_min;
    traj_Out.T_max=T_max;




end
