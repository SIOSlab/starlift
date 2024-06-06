clear
clc
close all



%% For Max's system

addpath(['G:\My Drive\College\Grad school\ME Aerospace\Project\V4 code (11 06 2023)\MDOtoolbox\MDOrun\'])
addpath(['G:\My Drive\College\Grad school\ME Aerospace\Project\V4 code (11 06 2023)\MDOtoolbox\modules'])


% [cost, time, mass] = MDOrun(thrustertype,proppower,n_thrusters,propellant,insulation_thk,surface_finish,debris_removed)
% [cost, time, mass] = MDOrun(1, 1000, 1, .01, .005, 2)

%using optimal results from OPTga.m
[cost, time, mass] = MDOrun(1, 7875, 1, .001, .2, 10.9144)

%test parameters: thruster type, power, propellant, insulation thickness, %surface finish, debris removed
%thruster type: 1 - hall; 2 - DC ion
%power range:
%propellant type: 1 - Xe; 2 - Kr; 3 - Ar
%insulation thickness range:
%surface finish range:
%debris removed: 1 - 22