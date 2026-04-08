%% Max

% Estimates mass given other input parameters, estimated from
% https://ntrs.nasa.gov/api/citations/19920019663/downloads/19920019663.pdf
% Note that article was written in 1990s

% University of Alabama, Huntsville - Spacecraft Propulsion System Sizing
% Tool

function mass_out = MASS() % NEED TO FIGURE OUT INPUTS
% Need to figure out where these variables go for inputs:

% Power
panelGeo = 1; % solar panel geometry, fixed or spinning
panelType = 1; % mass of fixed solar arrays depends heavily on panel type. Article cites diffrent thicknesses of Si or GaAs
panelTypeMultiplier = [0.0230 0.0194 0.0156]; % multiplier from article page 2-12 to 2-13
powerEOL = 15000; % [W], steady-state power usage required at spacecraft end of life
tMission = 6; % [years], mission duration
VBus = 60; % [volts], voltage that the main bus uses

% ADCS
ADCSCat = 1; % ADCS category, defined by article on page 3-10. Goes from 1 to 3
boreAccuracy = 0.24; %degrees
torqueWorst = 0.0003; % Nm, worst case cyclic disturbance
isAutoTrack = 0; % boolean, says whether or not RF autotracking is included

% Payload


% Structures
compMakeup = .75; % percentage makeup of structure being composites

%% Power. double check units, some expressions in article require W while others require kW
lifeFactor = 0.9622 * tMission;

% Mass of solar array
if panelGeo == 1 % if solar panels are fixed and not spinning
    mPanel = 18.3 + panelTypeMultiplier(panelType) * (powerEOL / lifeFactor);
else
    mPanel = 0.066 * (0.962 * tMission ^ -0.065) * powerEOL;
end

% Mass of battery
mBattery = 1.0943 + 0.0719 * powerEOL; % assuming NiCd, which is outdated

% Mass of PPU
mPPU = (0.173 + 0.01856 * powerEOL * 10^-3) * (mPanel + mBattery);

% Mass of SADA
allowableVoltages = [28 42 120]; % standardized voltages, as per article page 2-18
closestVoltage = find(min(abs(allowableVoltages - VBus)));

switch closestVoltage %ONLY ASSUMING THREE POSSIBLE BUS VOLTAGES. DOUBLE CHECK THIS
    case 1 % 28V bus
        if powerEOL < 3000
            mSADA = 9.5860 + 4.5600e-4 * powerEOL;
        elseif powerEOL < 6500
            mSADA = 11.666 + 4.9262e-4 * powerEOL;
        else
            mSADA = 12.768 + 6.1657e-4 * powerEOL;
        end
    case 2 % 42V bus
        if powerEOL < 4500
            mSADA = 9.6950 + 2.6333e-4 * powerEOL;
        else
            mSADA = 11.680 + 3.2855e-4 * powerEOL;
        end
    case 3 % 120V bus
        mSADA = 9.6898 + 8.6561e-5 * powerEOL;
end

mPower = mPanel + mBattery + mPPU + mSADA;

%% ADCS
mMomWheel = 2.1378 * boreAccuracy ^ -1.7529;
mRWA = 576.26 * torqueWorst ^ 0.35800;
switch ADCSCat
    case 1
        mADCS = 2.0 + 0.3 + 7.0 + 0.2;
    case 2
        mADCS = 3.0 + 0.6 + 1.4 + 5.0 + mMomWheel + mRWA;
    case 3
        mADCS = 3.0 + .6 + 5.0 + 14.0 + mRWA;
end

if isAutoTrack == 1 % checks whether or not to add autotracker
    mAutoTrack = 6.0 + 5.0 + 4.5;
else
    mAutoTrack = 0;
end

%% Payload

%% Structure
mFracStruct = (-24 * compMakeup) + (0.0016 * compMakeup)

end