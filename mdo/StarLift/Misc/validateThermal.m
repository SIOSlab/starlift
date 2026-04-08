parameter=MDOgenerateParamater()

thrustertype=2;
proppower=3000;
insulation_thk=.01;
surface_finish=0.2;
debris_removed=4;


parameter.T_E_cold= 273;
parameter.T_E_hot= 273+40


design_variables.disc_power = 16;
design_variables.thruster_type = thrustertype;
design_variables.insulation_thk = insulation_thk;
design_variables.FinishType = surface_finish;
design_variables.debris_removed = ceil(debris_removed); %NOTE


parameter=MDOgenerateParamater();

traj_Out=MDOtrajectory(parameter);
thermal_Out=MDOthermal(parameter,design_variables,traj_Out)
% power_Out = MDOpower(design_variables,traj_Out,thermal_Out,parameter);
% propulsion_Out= MDOpropulsion(design_variables,traj_Out, power_Out, thermal_Out, parameter);
% cost_Out = MDOcost(design_variables,propulsion_Out, power_Out, thermal_Out, parameter);
% 
% 
% mass = propulsion_Out.m_wet;
% time = propulsion_Out.mission_time/3600/24/365;
% cost = double(cost_Out.total_cost);
% 
% if time > 10
%     cost = cost+cost*(time-10);
