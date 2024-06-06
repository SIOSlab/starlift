%% Table testing
% Created by Max Luo, 3/20/24
% Last edited by Max Luo, 3/20/24
% This script has no function for MDO, and served only as a test bed to
% test native MATLAB functionality for interpreting Excel sheets. Future
% developers of Starlift should come back to this file if functionality is
% changed or broken.

clear all
clc
close all

excelPath = "G:\My Drive\College\Grad school\ME Aerospace\Project\Luo code ACTIVE\Misc\Requirements library\REQUIREMENTS ACTIVE";
addpath(excelPath);
T = readtable("SPECIFY_PARAMETERS_ACTIVE.xlsx");
T2 = readtable("SPECIFY_PARAMETERS_ACTIVE.xlsx",'Sheet','propChoice');

S = table2struct(T);
S2 = table2struct(T2);

parameter.mue=3.986004418e14; %gravitational constant [m3 s−2]
parameter.g = 9.81; %g, m/s^2
parameter.r_e=6.3781e6;% radius of Earth, 
parameter.sigma=5.699e-8; %Stefan-Boltzman Constant, W/m^2 K^4
parameter.R_univ   = 8.314; %universal gas constant,    

for i = 1:length(S)
    if isempty(S(i).OVERRIDE) || isnan(S(i).OVERRIDE)
        parameter.(S(i).MATLABFIELDNAME) = S(i).DEFAULT;
    else
        parameter.(S(i).MATLABFIELDNAME) = S(i).OVERRIDE;
    end
end