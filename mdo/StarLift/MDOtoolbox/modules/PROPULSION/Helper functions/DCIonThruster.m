%% DCIonThruster
% Created by Max Luo, 4/24/24
% Last edited by Max Luo, 4/24/24
% This function serves to model the performance of a DC Ion Thruster
% subject % to a specific propellant and operating power. All numbers are
% taken from % Hofheins et al.'s debris removal mission, which in turn is
% based on Petro and Sedwick 2016.
%
% Inputs: operating power[W], propellant molar mass[g/mol]
% Outputs: thrust[N], Isp[s]

function [thrust, Isp] = DCIonThruster(power, propMass)

MXe = 131.3; % g/mol, reference molar mass of Xe

thrust = sqrt(propMass/MXe)*(power) * 33.98 * 1e-6; % [N], from Petro and Sedwick
Isp = sqrt(propMass/MXe) * (360 * log(power/1e3)+3008); % this doesn't seem right - why is power being divided by 1000?


end