%SYSEN 5350 - Multidisciplinary Design Optimization
%Orbit Calculation

clear all
close all

%specify how many steps to loop
n = 1000;

%% Design Variables Vector Input
ri = 6.61534*10^6; %[m] Initial Orbital Radius
mstruct = 1655.48; %[kg] Initial Spacecraft Structure Mass (without propellant)
Isp = 4733.78; %[s] Specific Impulse for low thrust continuous orbit raise

%% Parameters
deltav = 1000/n; %[m/s] Total Change in Velocity the spacecraft should do
g = 9.81; %[m/s^2] Initial Gravitational Acceleration
rE = 6378100; %[m] Radius of Earth
G = 6.67408*10^-11; %[m^3/(kg*s^2)] Gravitational Constant
mE = 5.972*10^24; %[kg] Mass of Earth
cD = 0.1; %[-] spacecraft coefficient of drag
A = 1; %[m^2] Frontal area of spacecraft
ti = 0; %[s] initial time of simulation
Pe = 1*10^-5; %[kPa] spaceraft thruster plume's pressure
Ae = 1; %[m^2] spacecraft thruster nozzle's exit area
Ve = 1000; %[m/s] Spacecraft thruster nozzle plume's exit velocity %citation https://electricrocket.org/2019/590.pdf
eta = 2; %[-] Thrust multiplier

%% Set initial time and orbit radius 
    i = 1;
    ri(i) = ri;
    ti(i) = ti;

%% Calculate every mass from final to initial
    hi(n) = ri - rE; %[m] initial height of spacecraft above Earth
    gi = g./((1 + hi(n)./rE).^2); %[m/s^2] initial gravitational acceleration on spacecraft
    ms(n) = mstruct;
    mp(n) = ms(n).*(exp(deltav./(Isp.*gi))-1);
    mi(n) = ms(n) + mp(n);
    mf(n) = mi(n) - mp(n);
    for i = 1:(n-1)
        ms(n-i) = ms(n+1-i) + mp(n+1-i);
        mp(n-i) = ms(n+1-i).*(exp(deltav./(Isp.*gi))-1);
        mi(n+1-i) = ms(n+1-i) + mp(n+1-i);
        mf(n+1-i) = mi(n+1-i) - mp(n+1-i);
    end

%% Calculate first step
    i = 1;

    % Gravity Module
    hi(i) = ri(i) - rE; %[m] initial height of spacecraft above Earth
    gi(i) = g./((1 + hi(i)./rE).^2); %[m/s^2] initial gravitational acceleration on spacecraft
    
    % Propellant Module
    mi(i) = mp(i) + ms(i); %[kg] initial spacecraft mass (propellant and structure)
    mf(i) = mi(i) - mp(i); %[kg] final spacecraft mass
    
    %Low Altitude Atmosphere Module (NASA Model)
    Ti(i) = -131.21 + 0.00299.*hi(i); %[deg C] Initial atmospheric temp (NASA model)
    Pi(i) = 2.488.*(((Ti(i) + 273.1)./216.6).^-11.388); %[kPa] Initial atmospheric pressure (NASA model)
    rhoi(i) = Pi(i)./(0.2869.*(Ti(i)+273.1)); %[kg/m^3] Initial atmospheric density (NASA model)
    
    %Low Altitude Station-Keeping Module
    Fgi(i) = G.*mE.*mi(i)./ri(i).^2; %[N] Initial force of gravity on the spacecraft
    Fci(i) = Fgi(i); %[N] Initial centrifugal force on the spacecraft (circular orbit) (CONSTRAINT)
    Vi(i) = sqrt(Fci(i).*ri(i)./mi(i)); %[m/s] Initial spacecraft velocity
    Fdi(i) = (1/2).*rhoi(i).*Vi(i).^2.*cD.*A; %[N] Initial force of drag on spacecraft
    Fthi(i) = Fdi(i); %[N] initial thrust of spacecraft needed for station-keeping
    
    %Orbit Raising Module
    Ftho(i) = eta.*Fthi(i); %[N] orbit raising thrust multiplier
    tf(i) = ti(i) + (mi(i) - mf(i))./((Ftho(i) - (Pe - Pi(i)).*Ae.*1000)./Ve); %[s] Time taken to hit max orbit raise
    rf(i) = Vi(i).*(tf(i)-ti(i)) + ri(i); %[m] final orbital radius of spacecraft
    hf(i) = rf(i) - rE; %[m] final height of spacecraft above Earth
    gf(i) = g./((1 + hf(i)./rE).^2); %[m/s^2] final gravitational acceleration on spacecraft
    
    %High Altitude Atmosphere Module (NASA Model)
    Tf(i) = -131.21 + 0.00299.*hf(i); %[deg C] Final atmospheric temp (NASA model)
    Pf(i) = 2.488.*(((Tf(i) + 273.1)./216.6).^-11.388); %[kPa] final atmospheric pressure (NASA model)
    rhof(i) = Pf(i)./(0.2869.*(Tf(i)+273.1)); %[kg/m^3] final atmospheric density (NASA model)
    
    % High Altitude Station-Keeping Module
    Fgf(i) = G.*mE.*ms(i)./rf(i).^2; %[N] final force of gravity on the spacecraft
    Fcf(i) = Fgf(i); %[N] final centrifugal force on the spacecraft (circular orbit) (CONSTRAINT)
    Vf(i) = sqrt(Fcf(i).*rf(i)./mf(i)); %[m/s] final spacecraft velocity
    Fdf(i) = (1/2).*rhof(i).*Vf(i).^2.*cD.*A; %[N] final force of drag on spacecraft
    Fthf(i) = Fdf(i); %[N] final thrust of spacecraft needed to station-keep circular orbit
    Ftot(i) = Fthi(i) + Ftho(i) + Fthf(i); %[N] Total thrust to complete orbit raise (OBJECTIVE minimize)

%% Calculate every step from final to initial
for i = 2:(n)

    %Connect to next propagation
    ti(i) = tf(i-1); %loop time
    ri(i) = rf(i-1); %loop orbital radius
    hi(i) = hf(i-1); %loop height
    gi(i) = gi(i-1); %loop gravitational acceleration
    Ti(i) = Tf(i-1); %loop temp
    Pi(i) = Pf(i-1); %loop pressure
    rhoi(i) = rhof(i-1); %loop density
    Ftho(i) = Ftho(i-1); %[N] orbit raising thrust multiplier
    Fgi(i) = Fgf(i-1); %loop gravity force
    Fci(i) = Fcf(i-1); %loop centrifugal force
    Vi(i) = Vf(i-1); %loop velocity
    Fdi(i) = Fdf(i-1); %loop drag force
    Fthi(i) = Fthf(i-1); %loop thrust force

    tf(i) = ti(i) + (mi(i) - mf(i))./((Ftho(i) - (Pe - Pi(i)).*Ae.*1000)./Ve); %[s] Time taken to hit max orbit raise
    rf(i) = Vi(i).*(tf(i) - ti(i)) + ri(i); %[m] final orbital radius of spacecraft
    hf(i) = rf(i) - rE; %[m] final height of spacecraft above Earth
    gf(i) = g./((1 + hf(i)./rE).^2); %[m/s^2] final gravitational acceleration on spacecraft

    %High Altitude Atmosphere Module (NASA Model)
    Tf(i) = -131.21 + 0.00299.*hf(i); %[deg C] Final atmospheric temp (NASA model)
    Pf(i) = 2.488.*(((Tf(i) + 273.1)./216.6).^-11.388); %[kPa] final atmospheric pressure (NASA model)
    rhof(i) = Pf(i)./(0.2869.*(Tf(i)+273.1)); %[kg/m^3] final atmospheric density (NASA model)

    % High Altitude Station-Keeping Module
    Fgf(i) = G.*mE.*ms(i)./rf(i).^2; %[N] final force of gravity on the spacecraft
    Fcf(i) = Fgf(i); %[N] final centrifugal force on the spacecraft (circular orbit) (CONSTRAINT)
    Vf(i) = sqrt(Fcf(i).*rf(i)./mf(i)); %[m/s] final spacecraft velocity
    Fdf(i) = (1/2).*rhof(i).*Vf(i).^2.*cD.*A; %[N] final force of drag on spacecraft
    Fthf(i) = Fdf(i); %[N] final thrust of spacecraft needed to station-keep circular orbit
    Ftot(i) = Fthi(i) + Ftho(i) + Fthf(i); %[N] Total thrust to complete orbit raise (OBJECTIVE minimize)
end

%% Visualization

time = [ti tf];
radius = [ri/1000 rf/1000];
plot(time, radius,'*')

set(gca,'fontname','times')
title('Orbit Raise Height vs. Time')
xlabel('Time to Complete Orbit Raise [s]')
ylabel('Orbital Radius [km]')
hold off

%% Print my workspace variable values 
myvals = whos;
for n = 1:length(myvals)
       myvals(n).name
       eval(myvals(n).name)
end