%% Setup the Problem:
clear 
%clc
close all
format longE
options = optimset('PlotFcns',@optimplotfval);
%proppower,insulation_thk,surface_finish,debris_removed
%% First, we Calculate the optimum of the unscaled case.

L=eye(4);

x0 = [2980, .053, 0.45, 4];
%x0=[5000,0.01,0.2,6];

func1 = @(x)(1e-8*scaled_MDOrun(1, x(1), x(2), x(3), x(4),L));


LB = [500,.001,.2,1];
UB = [5000,0.02,1,22];
options = optimset('PlotFcns',@optimplotfval);

figure
tic
[xbest_unscaled, ~] = NM_fminsearchbnd(func1,x0,LB, UB)
toc
[cost_unscaled,~,~] = scaled_MDOrun(1,xbest_unscaled(1),xbest_unscaled(2),xbest_unscaled(3),xbest_unscaled(4),L)


[~, Hessian]=MDOHessian(xbest_unscaled,L)

hessE=eig(Hessian);
param=max(abs(hessE))/min(abs(hessE))

 %L=[1/sqrt(abs(Hessian(1,1))) 0 0; 0 1/sqrt(abs(Hessian(2,2))) 0; 0 0 1/sqrt(abs(Hessian(3,3))) ]
 L=[1e0 0 0; 0 1e-4 0; 0 0 1e-1];
%% Next, We calculate the optimum of the scaled case. 
func2 = @(x)(1e-8*scaled_MDOrun(1, L(1,1)*x(1), L(2,2)*x(2), L(3,3)*x(3), x(4),L));

x0 = [2980/L(1,1), .053/L(2,2), 0.45/L(3,3), 4];
%x0 = [5000/L(1,1), .01/L(2,2), 0.2/L(3,3), 6];
LB = [500/L(1,1),.001/L(2,2),.2/L(3,3),1];
UB = [5000/L(1,1),0.02/L(2,2),1/L(3,3),22];


figure

tic
[xbest_scaled, fval1] = NM_fminsearchbnd(func2,x0,LB, UB);
toc
[cost_scaled,time,mass] = scaled_MDOrun(1,L(1,1)*xbest_scaled(1),L(2,2)*xbest_scaled(2),L(3,3)*xbest_scaled(3),xbest_scaled(4),L);


[~, Hessian_scaled]=MDOHessian(xbest_scaled,L)

hessE=eig(Hessian_scaled);
param=max(abs(hessE))/min(abs(hessE))

xbest_adapted=[xbest_scaled(1)*L(1,1); xbest_scaled(2)*L(2,2); xbest_scaled(3)*L(3,3); xbest_scaled(4)]
Hessian
