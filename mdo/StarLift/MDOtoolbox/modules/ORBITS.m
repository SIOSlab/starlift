%% Orbits
% Created by Max Luo, 4/24/24
% Last edited by Max Luo, 4/24/24
% This function takes in the MDO parameter, a starting orbit, and an ending
% orbit, and offers a delta-V estimate based either on an impulsive burn or
% low-thrust burn, plus the time required to execute the maneuver.
% Currently, only circular orbits are modeled.
%
% altStart, altEnd (m)

function orbits_out = ORBITS(altStart, altEnd, parameter)

% Convert from altitude to radius
rStart = altStart + parameter.r_e;
rEnd = altEnd + parameter.r_e;

% dV calculations symmetric for circular orbits
rLow = min([rStart rEnd]);
rHigh = max([rStart rEnd]);

%% Hohmann transfer
aTransfer = (rLow + rHigh) / 2; % semimajor axis of transfer ellipse
vLow = sqrt(parameter.mue * (1/rLow)); % velocity of lower altitude circle
vLowTransfer = sqrt(parameter.mue * (2/rLow - 1/aTransfer)); % velocity at transfer ellipse periapsis
vHighTransfer = sqrt(parameter.mue * (2/rHigh - 1/aTransfer)); % velocity at transfer ellipse apoapsis
vHigh = sqrt(parameter.mue * (1/rHigh)); % velocity of higher altitude circle

dVLow = abs(vLow - vLowTransfer);
dVHigh = abs(vHigh - vHighTransfer);
dVHohmann = dVLow + dVHigh;

timeHohmann = pi * sqrt(aTransfer^3 / parameter.mue); % transfer time of Hohmann is approximately just half of an ellipse

%% Low thrust transfer
dVConstThrust = abs(sqrt(parameter.mue/rLow) - sqrt(parameter.mue/rHigh));


%% Putting it all together
orbits_out.dVHohmann = dVHohmann;
orbits_out.timeHohmann = timeHohmann;
orbits_out.dVConstThrust = dVConstThrust;

end