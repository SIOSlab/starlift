% Arthur Chadwick
% SYSEN 5350 HW3 Part A Q2
% 3/19/24

clear all
close all

% We started with the SAdemo0.m files for the atom configuration exercise we
% did in class and then adapted them to meet our design optimization problem.

% Ro = [ri mstruct Isp]
% Initial Orbital Radius ri [m]
% Mass of Spacecraft Structure mstruct [kg]
% Thruster Specific Impulse Isp [s]

Ro=[6571000 1000 3000];   % Random Initial Design Guess

% Evaluate Energy, or in our case Mass, for the Initial Design
[E]=evalAtomsThrust(Ro);
disp(['Initial Energy: ' num2str(E)])

% Input Matlab Files for Simulated Annealing
file_eval='evalAtomsThrust';
file_perturb='perturbAtomsThrust';
    To=100; options(1)=To;
    schedule=2; options(2)=schedule;
    dT=0.75; options(3)=dT;
    neq=50; options(4)=neq;
    nfrozen=3; options(5)=nfrozen;
    diagnostics=0; options(6)=diagnostics;
    plotflag=1; options(7)=plotflag; 

tic;
[Rbest,~,Rhist]=SA(Ro,file_eval,file_perturb,options);
CPUtime=toc

if plotflag
[Ebest]=evalAtomsThrust(Rbest(1,:))
end