function sensitivity=MDOsensitivityAnalysis(xbest)
%(thrustertype,proppower,insulation_thk,surface_finish,debris_removed)
% propPower=xbest(1);
% insulation_thk=xbest(2);
% surface_finish=xbest(3);
% debris_removed=xbest(4);


[cost0,~,~] = NM_MDOrun(xbest(1),xbest(2),xbest(3),xbest(4),xbest(5),xbest(6));
step=1e-4;

%Sensitivity Analyis of x1
perturbx1=xbest(1)*step;
[costx1High,~,~] = NM_MDOrun(1,xbest(1)+perturbx1,xbest(2),xbest(3),xbest(4));
%[costx1Low,~,~] = NM_MDOrun(1,xbest(1)-perturbx1,xbest(2),xbest(3),xbest(4));
gradX1=(costx1High-cost0)/(perturbx1);
sensitivityX1=xbest(1)/cost0*gradX1;

%Sensitivity Analyis of x2
perturbx2=xbest(2)*step;
[costx2High,~,~] = NM_MDOrun(1,xbest(1),xbest(2)+perturbx2,xbest(3),xbest(4));
%[costx2Low,~,~] = NM_MDOrun(1,xbest(1),xbest(2)-perturbx2,xbest(3),xbest(4));
gradX2=(costx2High-cost0)/(perturbx2);
sensitivityX2=xbest(2)/cost0*gradX2;

%Sensitivity Analyis of x3
perturbx3=xbest(3)*step;
[costx3High,~,~] = NM_MDOrun(1,xbest(1),xbest(2),xbest(3)+perturbx3,xbest(4));
%[costx3Low,~,~] = NM_MDOrun(1,xbest(1),xbest(2),xbest(3)-perturbx3,xbest(4));
gradX3=(costx3High-cost0)/(perturbx3);
sensitivityX3=xbest(3)/cost0*gradX3;


%Sensitivity Analyis of x4
perturbx4=1;
[costx4High,~,~] = NM_MDOrun(1,xbest(1),xbest(2),xbest(3),xbest(4)+perturbx4);
%[costx3Low,~,~] = NM_MDOrun(2,xbest(1),xbest(2),xbest(3),xbest(4)-perturbx4);
gradX4=(costx4High-cost0)/(perturbx4);
sensitivityX4=xbest(4)/cost0*gradX4;

sensitivity.X=[sensitivityX1,sensitivityX2,sensitivityX3,sensitivityX4];

%Sensitivity Analysis of r_Low
parameter=MDOgenerateParamater();
perturbrLow=step*parameter.r_low;
parameter.r_low=parameter.r_low+perturbrLow;
[costR_low,~,~] = NM_MDOrunParameterSens(1,xbest(1),xbest(2),xbest(3),xbest(4),parameter);
gradR_low=(costR_low-cost0)/(perturbrLow);
sensitivityR_Low=parameter.r_low/cost0*gradR_low;

%Sensitivity Analysis of r_high
parameter=MDOgenerateParamater();
perturbr_high=step*parameter.r_high;
parameter.r_high=parameter.r_high+perturbr_high;
[costR_high,~,~] = NM_MDOrunParameterSens(1,xbest(1),xbest(2),xbest(3),xbest(4),parameter);
gradR_high=(costR_high-cost0)/(perturbr_high);
sensitivityR_High=parameter.r_high/cost0*gradR_high;

%Sensitivity Analysis of r_bus

parameter=MDOgenerateParamater();
perturbr_bus=step*parameter.r_bus;
parameter.r_bus=perturbr_bus+parameter.r_bus; %Spacecraft Bus Radius
[costr_bus,~,~] = NM_MDOrunParameterSens(1,xbest(1),xbest(2),xbest(3),xbest(4),parameter);
gradR_Bus=(costr_bus-cost0)/(perturbr_bus);
sensitivityR_Bus=parameter.r_bus/cost0*gradR_Bus;


sensitivity.P=[sensitivityR_Low, sensitivityR_High, sensitivityR_Bus];
end
