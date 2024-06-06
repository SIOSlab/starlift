function [cost,time,mass] = MDOrun(thrustertype,proppower,insulation_thk,surface_finish,debris_removed)

design_variables.disc_power = proppower;
design_variables.thruster_type = thrustertype;
design_variables.insulation_thk = insulation_thk;
design_variables.FinishType = surface_finish;
design_variables.debris_removed = ceil(debris_removed); %NOTE

parameter=MDOgenerateParamater();

traj_Out=TRAJECTORY(parameter);
thermal_Out=THERMAL(parameter,design_variables,traj_Out);
power_Out = POWER(design_variables,traj_Out,thermal_Out,parameter);
propulsion_Out= PROPULSION(design_variables,traj_Out, power_Out, thermal_Out, parameter);
cost_Out = COST(design_variables,propulsion_Out, power_Out, thermal_Out, parameter);


mass = propulsion_Out.m_wet;
time = propulsion_Out.mission_time/3600/24/365;
cost = double(cost_Out.total_cost);

if time > 10
    cost = cost+cost*(time-10);
end

if time == 0
    cost = 1e22;
end


%constraints
% maxTime = 4.2;
% if time >  maxTime
%     cost = cost*1e11;
% end
% 
% if design_variables.disc_power < 500
%     cost = cost*1e11;
% end
% 
% if design_variables.disc_power > 3000
%     cost = cost*1e11;
% end
% 
% if design_variables.debris_removed < 0
%     cost = cost*1e11;
% end
% 
% if design_variables.debris_removed > 22
%     cost = cost*1e11;
% end
% 
% if design_variables.insulation_thk <= 0
%     cost = cost*1e11;
% end
% 
% if design_variables.insulation_thk > .1
%     cost = cost*1e11;
% end
% 
% 
% end 