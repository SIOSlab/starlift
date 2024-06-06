%% Alex
function thermal_Out=THERMAL(parameter,design_variables)

%% outputs from trajectory block
eclipse_ratio_min = parameter.eclipse_ratio_min;
eclipse_ratio_max = parameter.eclipse_ratio_max;
% T_min=traj_Out.T_min; %[s]
% T_max=traj_Out.T_max; %[s]

%% extraction of design variables
L=design_variables.insulation_thk; %[m]
Pprop = design_variables.disc_power; %[W]

%% parameters
   %Temperature Limits: -10 to 50 C with 10C slack
T_E_cold=parameter.T_E_cold; %Minimum allowable electronics temperature, K
T_E_hot=parameter.T_E_hot; %Maximum allowable electronics temperature, K
k=parameter.k; %Thermal conductivity, W/mK
alpha_R=parameter.alpha_R; %Absorptance of Finish, Diffuse Quartz
epsilon_R=parameter.epsilon_R; %Emitance of Finish, Diffuse Quartz
r=parameter.r_bus; %Spacecraft Bus Radius
sigma=parameter.sigma; %Stefan-Boltzman Constant, W/m^2 K^4
q_EarthIR_Inc_low=parameter.q_EarthIR_Inc_low;
q_EarthIR_Inc_high=parameter.q_EarthIR_Inc_high;
q_albedo_Inc_low_60=parameter.q_albedo_Inc_low_60; %When beta=60 deg, 0 when beta=90
q_albedo_Inc_high_60=parameter.q_albedo_Inc_high_60; %When beta=60 deg, 0 when beta=90
q_solar_Inc_low_60=1414*(1-eclipse_ratio_max); %When beta=60 deg
q_solar_Inc_high_60=1414*(1-eclipse_ratio_min); %When beta=60 deg
q_solar_Inc_low_90=parameter.q_solar_Inc_low_90; %When beta=90 deg
q_solar_Inc_high_90=parameter.q_solar_Inc_high_90; %When beta=90 deg

epsilon_I=0.85;
alpha_I=abs(rem(design_variables.FinishType,1));


%% Calculations

%calculate power consumption of spacecraft, assuming that the
%extra-propulsion subsystem accounts for 5% of the power to the propulsion

Power_used=.05*Pprop+Pprop; %changed from Alex's original, to keep consistent with power block

%calculate surface area and cross section of spacecraft bus assuming
%spherical body
A_s=4*pi*r^2; %Spacecraft Bus Surface Area
A_c=pi*r^2; %Spacecraft Cross Sectional Area


q_IR=[q_EarthIR_Inc_low;q_EarthIR_Inc_high;q_EarthIR_Inc_low;q_EarthIR_Inc_high];
q_albedo=[q_albedo_Inc_low_60;q_albedo_Inc_high_60;0;0];
q_solar=[q_solar_Inc_low_60;q_solar_Inc_high_60;q_solar_Inc_low_90;q_solar_Inc_high_90];


q_env=alpha_I*q_solar + alpha_I*q_albedo + epsilon_I*q_IR; %Order: [Low_60, High_60, Low_90, High_90];


Q_elec=.05*Power_used; %Estimated heat loss due to internal resistance from battery and electronics


q_cold=min(q_EarthIR_Inc_high);
%q_cold=min(q_env)
[q_hot,I]=max(q_env);



%syms T_E T_S z
% syms z
% 
% eqn1=epsilon_I*sigma*T_S^4==k*(T_E-T_S)/L;
% eqn2=subs(eqn1,T_E,T_E_cold);
% eqn3=subs(eqn1,T_E,T_E_hot);
%T_S_cold=vpasolve(eqn2,T_S,[0 Inf])
%T_S_cold=double(root(L*epsilon_I*sigma*z^4 + k*z - T_E_cold*k, z));
T_CRoots=[L*epsilon_I*sigma, 0, 0, k, (- T_E_cold*k)];
T_S_cold=double(roots(T_CRoots));
%T_S_hot=vpasolve(eqn3,T_S,[0 Inf]);
%T_S_hot = double(T_S_hot)

% T_S_hot=double(root(L*epsilon_I*sigma*z^4 + k*z - T_E_hot*k, z));

T_HRoots=[L*epsilon_I*sigma, 0, 0, k, (- T_E_hot*k)];
T_S_hot=double(roots(T_HRoots));

T_S_cold=T_S_cold(T_S_cold>=0);
T_S_cold=T_S_cold(T_S_cold == real(T_S_cold));
T_S_hot=T_S_hot(T_S_hot>=0);
T_S_hot=T_S_hot(T_S_hot == real(T_S_hot));

% T_S_cold1=T_S_cold1(T_S_cold1>=0);
% T_S_cold1=T_S_cold1(T_S_cold1 == real(T_S_cold1))
% T_S_hot1=T_S_hot1(T_S_hot1>=0);
% T_S_hot1=T_S_hot1(T_S_hot1 == real(T_S_hot1))

if (~isempty(T_S_hot) && ~isempty(T_S_cold) && isreal(T_S_cold) && isreal(T_S_hot)) 

    T_S_hot=T_S_hot(1);
    T_S_cold=T_S_cold(1);

    A_radiator=-(- A_s*epsilon_I*sigma*T_S_hot.^4 + Q_elec + A_c*(alpha_I*q_albedo(I) + epsilon_I*q_IR(I)) + A_c*alpha_I*q_solar(I))/(- epsilon_R*sigma*T_E_hot.^4 + alpha_R*q_albedo(I) + epsilon_R*q_IR(I));
    
    
    if A_radiator<0
        A_radiator=0;
    end
    
    %Q_in_hot=Q_elec+ A_c* ( (q_solar(I)+q_albedo(I))*alpha_I+q_IR(I)*epsilon_I ) + A_radiator*(q_albedo(I)*alpha_R+q_IR(I)*epsilon_R);
    
    Q_in_cold=Q_elec+A_c*q_cold*epsilon_I  + A_radiator*q_cold*epsilon_R;
    
    
    
    
    A_insulation=A_s;
    V_insulation=A_insulation*L;
    m_insulation=50*V_insulation;
    
    
    
    Q_radiator=A_radiator.*epsilon_R*sigma*T_E_hot^4;
    m_radiator=1/55*Q_radiator;
    
    
    q_heat=1.25* (A_radiator*epsilon_R*sigma*T_E_cold.^4 + epsilon_I*sigma*A_s*T_S_cold.^4 - Q_in_cold);
    m_heater=0;
    P_heater=0;
    if q_heat>=0
        m_heater=q_heat.*.03/100;
        P_heater=q_heat.*1.05;
    
    end
    
    
    thermal_Out.thermal_power=double(P_heater);
    thermal_Out.mass_thermalsys=double(m_heater+m_radiator+m_insulation);
    thermal_Out.A_insulation = A_insulation;
    thermal_Out.m_radiator = m_radiator;
    thermal_Out.A_radiator=A_radiator;
    thermal_Out.m_heater=m_heater;
    thermal_Out.massfractionsThermal = [m_radiator, m_heater, m_insulation];
    
    %test1=epsilon*sigma*T_S_hot^4*A_radiator;
    %test2=q_hot*A_radiator+Q_electronics;
    
else
    %disp('Error: Invalid T')
    %fprintf('Broke at %d',L)
    thermal_Out.thermal_power=inf;
    thermal_Out.mass_thermalsys=inf;
    thermal_Out.A_insulation = inf;
    thermal_Out.m_radiator = inf;
    thermal_Out.A_radiator=inf;
    thermal_Out.m_heater=inf;

end

end