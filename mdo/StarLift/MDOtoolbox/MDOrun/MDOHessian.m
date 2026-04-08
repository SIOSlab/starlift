function [grad, Hessian]=MDOHessian(xbest,L)
%(thrustertype,proppower,insulation_thk,surface_finish,debriscaled_MDOruns_removed)
% propPower=xbest(1);
% insulation_thk=xbest(2);
% surface_finish=xbest(3);
% debris_removed=xbest(4);
L1=L(1,1);
L2=L(2,2);
L3=L(3,3);

[cost0,~,~] = scaled_MDOrun(1,L1*xbest(1),L2*xbest(2),L3*xbest(3),xbest(4),L);
step=1e-4;
%Sensitivity Analyis of x1
perturbx1=xbest(1)*step;
[costx1High,~,~] = scaled_MDOrun(1,L1*(xbest(1)+perturbx1),L2*xbest(2),L3*xbest(3),xbest(4),L);
[costx1Low,~,~] = scaled_MDOrun(1,L1*(xbest(1)-perturbx1),L2*xbest(2),L3*xbest(3),xbest(4),L);
gradX1=(costx1High-cost0)/(perturbx1);

%H11=(costx1High-2*cost0+costx1Low)/(perturbx1^2);
H11=centralDif(costx1High,costx1Low,cost0,perturbx1);

%sensitivityX1=xbest(1)/cost0*gradX1;
%Sensitivity Analyis of x2
perturbx2=xbest(2)*step;
[costx2High,~,~] = scaled_MDOrun(1,L1*xbest(1),L2*(xbest(2)+perturbx2),L3*xbest(3),xbest(4),L);
[costx2Low,~,~] = scaled_MDOrun(1,L1*xbest(1),L2*(xbest(2)-perturbx2),L3*xbest(3),xbest(4),L);
gradX2=(costx2High-cost0)/(perturbx2);

%H22=(costx2High-2*cost0+costx2Low)/(perturbx2^2);
H22=centralDif(costx2High,costx2Low,cost0,perturbx2);

%Sensitivity Analyis of x3
perturbx3=xbest(3)*step;
[costx3High,~,~] = scaled_MDOrun(1,L1*xbest(1),L2*xbest(2),L3*(xbest(3)+perturbx3),xbest(4),L);
[costx3Low,~,~] = scaled_MDOrun(1,L1*xbest(1),L2*xbest(2),L3*(xbest(3)-perturbx3),xbest(4),L);

%H33=(costx3High-2*cost0+costx3Low)/(perturbx3^2);
H33=centralDif(costx3High,costx3Low,cost0,perturbx3);
gradX3=(costx3High-cost0)/(perturbx3^2);
% sensitivityX3=xbest(3)/cost0*gradX3;


%Sensitivity Analyis of x4
% perturbx4=1;
% [costx4High,~,~] = scaled_MDOrun(1,xbest(1),xbest(2),xbest(3),xbest(4)+perturbx4,L);
% [costx4Low,~,~] = scaled_MDOrun(1,xbest(1),xbest(2),xbest(3),xbest(4)-perturbx4,L);
% gradX4=(costx4High-cost0)/(perturbx4);
% H44=centralDif(costx4High,costx4Low,cost0,perturbx4);

%Hessian=[H11 0 0 0 ; 0 H22 0 0; 0 0 H33 0; 0 0 0 H44];

Hessian=[H11 0 0 ; 0 H22 0 ; 0 0 H33];
grad=[gradX1;gradX2;gradX3];
% sensitivityX4=xbest(4)/cost0*gradX4;

%sensitivity.X=[sensitivityX1,sensitivityX2,sensitivityX3,sensitivityX4];
    function H =centralDif(upper,lower,middle,step)
        H=(upper-2*middle+lower)/(step^2);
    end


end
