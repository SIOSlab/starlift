function [xbest,Ebest,xhist]=SA(xo,file_eval,file_perturb,options);
% [xbest,Ebest,xhist]=SA(xo,file_eval,file_perturb,options);
%
% Single Objective Simulated Annealing (SA) Algorithm
%
% This function is a generic implementation of the single objective
% Simulated Annealing (SA) algorithm first proposed by Kirkpatrick,
% Gelatt and Vecchi. The algorithm tries to improve upon an initial
% configuration, xo, by evaluating perturbed configurations. When the
% system reaches the "frozen" state, the algorithm stops and the best
% configuration and search history are returned. The user can choose 
% from one of two cooling schedules: linear or exponential.
%
% Input:
% xo           initial configuration of the system (a row vector)      
% file_eval    file name (character string) of configuration evaluator;
%              assumes that E='file_eval'(x) is a legitimate function
%              call; set up function such that (scalar) output E will be
%              minimized.
% file_perturb file name (character string) of configuration perturbator;
%              assumes that xp='fname_perturb'(x) is a legitimate function
%              call. This function creates a "neighboring" configuration.
% options      algorithm option flags. Uses defaults, [ ], if left blank
%    (1)       To - initial system temperature - automatically determined if
%              left blank ([]). To should be set such that the expression
%              exp(-E(xo)/To)>0.99 is true, i.e. the initial system is "melted"
%    (2)       Cooling Schedule: linear=1, exponential=[2]
%    (3)       dT Temp. increment, e.g. [dT=0.9] for exp. cooling Tk=dT^k*To,
%              abs. temperature increment for linear cooling (Tk+1=Tk-dT);
%    (4)       neq = equilibrium condition, e.g. number of rearrangements
%              attempted to reach equilibrium at a given temperature, neq=[5]  
%    (5)       frozen condition - sets up SA exit criterion
%              nfrozen = non-integer, e.g. 0.1 SA interprets this numbers as Tmin,
%              the minimum temperature below which the system is frozen.
%              nfrozen = integer ,e.g. 1,2..  SA interprets this as # of successive 
%              temperatures for which the number of desired acceptances defined 
%              under options(4) is not achieved, default: nfrozen=[3]              
%    (6)       set to 1 to display diagnostic messages (=[1])
%    (7)       set to 1 to plot progress during annealing (=[0])
%
% Output:
% xbest        Best configuration(s) found during execution - row vector(s)
% Ebest        Energy of best configuration(s) (lowest energy state(s) found)
% xhist        structure containing the convergence history
%   .iter      Iteration number (number of times file_eval was called)
%   .x         current configuration at that iteration
%   .E         current system energy at that iteration
%   .T         current system temperature at that iteration
%   .k         temperature step index k
%   .C         specific heat at the k-th temperature
%   .S         entropy at the the k-th temperature
%   .Tnow      temperature at the k-th temperature step
%
% User Manual (article):   SA.pdf
%
% Demos:         SAdemo0 - four atom placement problem
%                SAdemo1 - demo of SA on MATLAB peaks function
%                SAdemo2 - demo of SA for Travelling Salesman Problem (TSP)
%                SAdemo3 - demo of SA for structural topology optimization
%                SAdemo4 - demo of SA for telescope array placement problem
%
% dWo,(c)  MIT 2004
%
% Ref: Kirkpatrick, S., Gelatt Jr., C.D. and Vecchi, M.P., "Optimization
% by Simulated Annealing", Science, Vol. 220, Number 4598, pp. 671-680, May
% 1983


%check input
if ~isempty(options)
    To=options(1);
    schedule=options(2);
    dT=options(3);
    neq=options(4);
    nfrozen=options(5);
    diagnostics=options(6);
    plotflag=options(7);
else
    % set all options to default
    % To - set initial system temperature
    eval(['Eo=' file_eval '(xo);']);
    To=abs(-Eo/log(0.99));   % set initial temperature such that probablility of 
    % accepting an inferior solution is initially equal to 0.99
    schedule=2;
    dT=0.9;   % this is the ratio dT=(T_i+1/T_i) for geometrical cooling
    neq=5;   % number of rearrangements accepted at a given T
    nfrozen=3;  % if neq hasn't been reached at nfrozen successive 
    % temperatures the system is considered frozen and the SA exits
    diagnostics=1; % display messages
    plotflag=0; %plot convergence 
end
%
nmax=neq*round(sqrt(max(size(xo)))); % nmax - maximum number of steps at one temperature, while
%                                      trying to establish thermal equilibrium
%
if nfrozen==round(nfrozen)
    % nfrozen is integer - look for nfrozen successive temperatures without
    % neq acceptances
    Tmin=0;
else
    Tmin=nfrozen; nfrozen=3;
end
  


% Step 1 - Show initial configuration
if diagnostics==1
disp('Initial configuration: ')
xo
end

% Step 2 - Evaluate initial configuration
eval(['Eo=' file_eval '(xo);']);
counter=1;
xnow=xo; Enow=Eo; nnow=1;
xhist(nnow).iter=counter;

xhist(nnow).x=xo;
xhist(nnow).E=Enow;
xhist(nnow).T=To;
%  still need to add .S         current entropy at that iteration
xbest=xnow;
Ebest=Enow;
Tnow=To;
if diagnostics==1
    disp(['Energy of initial configuration Eo: ' num2str(Eo)])
end

if plotflag
    figure(99)
    plot(counter,Enow,'k*');
    hold on
    plot(counter,Enow,'mo')
    xlabel('Iteration Number')
    ylabel('System Energy')
    title('SA convergence history')
    lastbest=counter;
    drawnow
end

frozen=0;  % exit flag for SA
naccept=1; % number of accepted configurations since last temperature change
Tlast=1;   % counter index of last temperature change
k=1;       % first temperature step
ET=[];     % vector of energies at constant system temperature

% start annealing
while (frozen<nfrozen)&(Tnow>Tmin)
    
%Step 3 - Perturb xnow to obtain a neighboring solution

if diagnostics
    disp(['Counter: ' num2str(counter) ' Temp: ' num2str(Tnow) ' Perturbing configuration'])
end

eval(['xp=' file_perturb '(xnow);']);

%Step 4 - Evaluate perturbed solution
eval(['Ep=' file_eval '(xp);'])
counter=counter+1;

%Step 5 - Metropolis Step

dE=Ep-Enow; % difference in system energy
PdE=exp(-dE/Tnow);
if diagnostics
    disp(['Counter: ' num2str(counter) ' Temp: ' num2str(Tnow) ' P(dE)= ' num2str(PdE)])
end

%Step 6 - Acceptance of new solution
if dE<=0 % energy of perturbed solution is lower , automatically accept
    nnow=nnow+1;
    xnow=xp; Enow=Ep; 
    xhist(nnow).iter=counter;
    xhist(nnow).x=xp; 
    xhist(nnow).E=Ep; 
    xhist(nnow).T=Tnow;
    naccept=naccept+1;
   if diagnostics
   disp(['Counter: ' num2str(counter) ' Temp: ' num2str(Tnow)  ' Automatically accept better configuration (downhill)'])
   end
   
else
   % energy of perturbed configuration is higher, but might still accept it
   randomnumber01=rand;
    if PdE>randomnumber01
        nnow=nnow+1;
        xnow=xp; Enow=Ep; 
        xhist(nnow).iter=counter;
        xhist(nnow).x=xp; 
        xhist(nnow).E=Ep;
        xhist(nnow).T=Tnow;
      if diagnostics
       disp(['Counter: ' num2str(counter) ' Temp: ' num2str(Tnow)  ' Accepted inferior configuration (uphill)'])
      end
      
    else
        % keep current configuration
        xnow=xnow;
        Enow=Enow;
      if diagnostics
       disp(['Counter: ' num2str(counter) ' Temp: ' num2str(Tnow)  ' Kept the current configuration'])
      end
    end
end
       ET=[ET; Enow];
       if plotflag
       figure(99)
       plot(counter,Enow,'k*');
       drawnow
       end


if Enow<Ebest
    % found a new 'best' configuration
    Ebestlast=Ebest;
    Ebest=Enow;
    xbest=xnow;
    if diagnostics
       disp(['Counter: ' num2str(counter) ' Temp: ' num2str(Tnow)  ' This is a new best configuration'])
    end
    if plotflag
       figure(99)
       plot(counter,Enow,'mo');
       plot([lastbest counter],[Ebestlast Enow],'m-');
       lastbest=counter;
       drawnow
    end
elseif Enow==Ebest
    same=0;
    for ib=1:size(xbest,1)
        if xbest(ib,:)==xnow
             if diagnostics
          disp(['Counter: ' num2str(counter) ' Temp: ' num2str(Tnow)  ' Found same best configuration']) 
             end
        same=1;
        end
    end
     
     if same ==0
       Ebestlast=Ebest;
       Ebest=Enow;
       xbest=[xbest ; xnow];
        if diagnostics
          disp(['Counter: ' num2str(counter) ' Temp: ' num2str(Tnow)  ' Found another best configuration'])
        end
      if plotflag
       figure(99)

       plot(counter,Enow,'mo');
       plot([lastbest counter],[Ebestlast Enow],'m-');
       lastbest=counter;
       drawnow
      end
    end
end

%Step 7 - Adjust system temperature
Told=Tnow;
if (naccept<neq)&(counter-Tlast)<nmax
    if diagnostics
       disp(['Counter: ' num2str(counter) ' Temp: ' num2str(Tnow)  ' Need to reach equilibrium at this temperature'])
    end
    % continue at the same system temperature
elseif (naccept<neq)&(counter-Tlast)>=nmax
    if diagnostics
       disp(['Counter: ' num2str(counter) ' Temp: ' num2str(Tnow)  ' System nearly frozen'])
    end
   
    Eavg=mean(ET);
    Evar=mean(ET.^2);
    C=(Evar-Eavg^2)/Tnow^2;     % specific heat
    S=log(nmax*length(unique(ET))/length(ET));
    xhist(k).k=k;
    xhist(k).C=C;
    xhist(k).S=S;
    xhist(k).Tnow=Tnow;
  
     
    frozen=frozen+1;
    Tlast=counter;
    naccept=0;
   
   
    switch schedule
        case 1
            % linear cooling
            Tnow=Tnow-dT;
              if Tnow<0
                  frozen=nfrozen; %system temperature cannot go negative, exit
              end
        case 2
            % exponential cooling
              Tnow=dT*Tnow;
        case 3
              Tindex=Tindex+1;
              if Tindex>size(Tuser,1)
                  frozen=nfrozen; % have run through entire user supplied cooling schedule
              else
              Tnow=Tuser(Tindex,1);
              neq=Tuser(Tindex,2);
              end
        otherwise
              disp('Erroneous cooling schedule choice - option(2) - illegal')
    end
    
   
    k=k+1;
     
   if plotflag
    figure(98)
    hist(ET); Nh=hist(ET); Nh=max(Nh);
    hold on
    plot([Eavg Eavg]',[0 Nh+1],'k--')
    text(Eavg, Nh+1, ['T=' num2str(Told,2)])
    xlabel('Energy')
    ylabel('Occurences')
    drawnow
  end
    ET=[];
    
 elseif (naccept==neq)
    if diagnostics
       disp(['Counter: ' num2str(counter) ' Temp: ' num2str(Tnow)  ' System reached equilibrium'])
    end
    
    Eavg=mean(ET);
    Evar=mean(ET.^2);
    C=(Evar-Eavg^2)/Tnow^2;     % specific heat
    S=log(nmax*length(unique(ET))/length(ET));
    xhist(k).k=k;
    xhist(k).C=C;
    xhist(k).S=S;
    xhist(k).Tnow=Tnow;
   
    
    Tlast=counter;
    naccept=0;
    
    switch schedule
        case 1
            % linear cooling
            Tnow=Tnow-dT;
              if Tnow<0
                  frozen=nfrozen; %system temperature cannot go negative, exit
              end
        case 2
            % exponential cooling
              Tnow=dT*Tnow;
        case 3
            % user supplied cooling
              Tindex=Tindex+1;
              if Tindex>size(Tuser,1)
                  frozen=nfrozen; %have run through entire user supplied cooling schedule
              else
              Tnow=Tuser(Tindex,1);
              neq=Tuser(Tindex,2);
              end
              
        otherwise
              disp('Erroneous cooling schedule choice - option(2) - illegal')
    end
  
 
     k=k+1;
     
  if plotflag
        figure(98)
    hist(ET); Nh=hist(ET); Nh=max(Nh);
    hold on
    plot([Eavg Eavg]',[0 Nh+1],'k--')
    text(Eavg, Nh+1, ['T=' num2str(Told,2)])
    xlabel('Energy')
    ylabel('Occurences')
    drawnow
end

  ET=[];
 end

end %while (frozen<nfrozen)&(Tnow>tmin)
    
% Reached end of SA
 if plotflag
       figure(97)
       k=k-1;
       for ind=1:k
           S(ind)=xhist(ind).S;
           C(ind)=xhist(ind).C;
           Tnow(ind)=xhist(ind).Tnow;
       end
       
           plot([1:k],C,'bo')
           hold on
           plot([1:k],S,'ms')
           plot([1:k],log(Tnow),'kd')
           legend('C-specific heat','S-entropy','ln(T)-temperature')
           xlabel('Temperature Step')
           title('Simulated Annealing Evolution')
           plot([1:k],C,'b-')
           plot([1:k],S,'m-')
           plot([1:k],log(Tnow),'k-')
           
           drawnow
           
  end
  
    if diagnostics
       disp(['Counter: ' num2str(counter) ' Temp: ' num2str(Tnow)  ' System frozen, SA ended'])
       disp(['Best configuration: '])
       xbest
       disp(['Lowest System Energy: ' num2str(Ebest) ])
    end