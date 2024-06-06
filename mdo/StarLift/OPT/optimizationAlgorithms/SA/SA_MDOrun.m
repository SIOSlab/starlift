function [cost,time,mass] = SA_MDOrun(X)
design_variables.thruster_type = X(1);
design_variables.disc_power = X(2);
design_variables.insulation_thk = X(3);
design_variables.FinishType = X(4);
design_variables.debris_removed = ceil(X(5)); %NOTE

parameter=MDOgenerateParamater();

traj_Out=MDOtrajectory(parameter);
thermal_Out=MDOthermal(parameter,design_variables,traj_Out);
power_Out = MDOpower(design_variables,traj_Out,thermal_Out,parameter);
propulsion_Out= MDOpropulsion(design_variables,traj_Out, power_Out, thermal_Out, parameter);
cost_Out = MDOcost(design_variables,propulsion_Out, power_Out, thermal_Out, parameter);


mass = propulsion_Out.m_wet;
time = propulsion_Out.mission_time/3600/24/365;
cost = double(cost_Out.total_cost);

if time > 10
    cost = cost+cost*(time-10);
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