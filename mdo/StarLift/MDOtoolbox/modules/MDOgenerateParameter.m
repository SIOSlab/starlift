% Last edited by Max Luo, 5/11/24
% MDOgenerateParameter handles the design variables and extra parameters
% as required by the MDO problem. To use this function properly, ensure you
% have access to your main library of parameters, which by default is in
% Library\"PARAMETERS_LIBRARY.xlsx".

% The "default" column should be any parameters you wish to hold constant
% in the absence of any specified override value. For example, if the
% majority of your solar arrays have an 

% optVars is a variable that holds the variables that are design variables,
% as opposed to parameters

function [parameter, optVars]=MDOgenerateParameter()
    %% Excel handling
    excelName = "PARAMETERS_LIBRARY.xlsx";
    sheetNames = sheetnames(excelName);

    T = readtable(excelName);
    S = table2struct(T);
    % creates a Table object from an Excel file, then immediately turns
    % that table into a struct

    T_prop = readtable(excelName, 'Sheet', sheetNames{2});
    S_prop = table2struct(T_prop);
    % table -> struct for choices of propellant

    %% Struct populating
    %%%%%% Immutables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                                                     %
    parameter.mue=3.986004418e14; %gravitational constant [m3 s−2]
    parameter.g = 9.81; %g, m/s^2
    parameter.r_e=6.3781e6;% radius of Earth, m
    parameter.sigma=5.699e-8; %Stefan-Boltzman Constant, W/m^2 K^4
    parameter.R_univ   = 8.314; %universal gas constant,               [J]
    %                                                                     %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % References, etc. that still need to be sorted (3/20/24)
    % battery density https://www.eaglepicher.com/markets/space/satellites/
    % panel areal density https://www.valispace.com/wp-content/uploads/2018/12/EPS-sizing-tutorial-1.pdf
    % Drag coefficient, worst case scenario SMAD
    % spacecraft density: https://www.researchgate.net/figure/Lifetime-of-a-spherical-satellite-with-mass-density-02-g-cm-3-versus-altitude-and-size_fig1_231941864#:~:text=altitude%20and%20size%2C%20assuming%20average,%5BLoftus%20and%20Reynolds%201993%5D.
    % thermal conductivity per https://iopscience.iop.org/article/10.1088/1757-899X/396/1/012061/pdf
    % Falcon9toSSO = 11e3;
    % FalconHeavytoSSO = 11e3*8/3.49;
    % FalconHeavyCost = 97e6;
    optVars = {};   

    for i = 1:length(S) % handles DEFAULT vs OVERRIDE behavior
        if isempty(S(i).OVERRIDE) || isnan(S(i).OVERRIDE)
            parameter.(S(i).MATLABFIELDNAME) = S(i).DEFAULT;
        else
            parameter.(S(i).MATLABFIELDNAME) = S(i).OVERRIDE;
        end
        if S(i).OPTTARGET == 1
            optVars{end + 1} = S(i).MATLABFIELDNAME;
        end
        %% if propflag ==1, do something else with it
        
        
    end

    for i = 1:length(S_prop)
        propChoices{i} = S_prop(i).PROPELLANT;
        propMolarMass(i) = S_prop(i).MOLARMASS;
    end

    parameter.propChoices = propChoices;
    parameter.propMolarMass = propMolarMass;

    %% Extra processing
    
    % Convert from km to m
    parameter.z1 = parameter.z1 * 1000;
    parameter.z2 = parameter.z2 * 1000;

    % Orbital radii (not altitude)
    parameter.r1 = parameter.r_e + parameter.z1;
    parameter.r2 = parameter.r_e + parameter.z2;

    % Calculating orbital period
    parameter.T1 = 2 * pi * sqrt(parameter.r1/parameter.mue);
    parameter.T2 = 2 * pi * sqrt(parameter.r2/parameter.mue);
end