% test parameters:(1, 20000, 1); (Hall, 20 kW, Xe)

function [cost,time,mass, massfractions, massfractionsSubsys, costfractions, costfractionsSubsys, costfractionsSC, propFtdrag]...
    = MDOrun(thrustertype,proppower,propellant)
design_variables.disc_power = proppower;
design_variables.thruster_type = thrustertype;
design_variables.propellant = propellant;

[parameter, optVars] =MDOgenerateParameter();

power_Out = POWER(design_variables, parameter);
propulsion_Out= PROPULSION(design_variables, power_Out, parameter);
cost_Out = COST(design_variables,propulsion_Out, power_Out, parameter);

mass = propulsion_Out.m_wet;
propFtdrag = propulsion_Out.Ftdrag;
massfractions = propulsion_Out.massfractions;
massfractionsSubsys = propulsion_Out.massfractionsSubsys;
costfractions = cost_Out.cost_fractions;
costfractionsSubsys = cost_Out.cost_fractionsSubsys;
costfractionsSC     = cost_Out.cost_spacecraft;
time = propulsion_Out.mission_time/3600/24/365;
cost = double(cost_Out.total_cost);

if time > 10
    cost = cost+cost*(time-10);
end

if time == 0
    cost = 1e22;
end
