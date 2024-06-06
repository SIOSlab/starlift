function [cost,time,mass] = scaled_MDOrun(thrustertype,proppower,insulation_thk,surface_finish,debris_removed,~)

designVars=[proppower,insulation_thk,surface_finish,debris_removed];%/L;


design_variables.disc_power = designVars(1);
design_variables.thruster_type = thrustertype;
design_variables.insulation_thk = designVars(2);
design_variables.FinishType = designVars(3);
design_variables.debris_removed = ceil(designVars(4)); %NOTE Ceil on debris removed



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
    cost = cost+cost*(time-5);
end

%cost=cost*1e-8; %Scale Cost

%constraints
% maxTime = 4.2;
% if time >  maxTime
%     cost = cost*1e11;
% % end
% if surface_finish<0.2
%     cost = cost*1e11;
% end
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


end 