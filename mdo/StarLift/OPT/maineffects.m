% doe plotter

%% 1
maineffectss =[81964738.48	238579480.7	88827579.51	1274202.862	14012885.74	1892413474];
titles = {'$n_{debris}$', '$\zeta$', '$P_{propulsion}$','$k_{propsys}$'};

barh(maineffectss)
yticklabels(titles)

%% 2
propsyseffects = [-46236519.14 ,35728219.34,0,0]/1e6;
powereffects = [0, -64626447.33, 112478029.7, -61475003.68]/1e6;
propellanteffects = [-26834217.99, 12339528.3, 49653833.22, 0]/1e6;
taueffects = [505688.9771, 131412.4539, -169803.6182, -467297.8128]/1e6;
alphaeffects = [4768432.41, 2238010.459, -1331510.444, -5674932.426]/1e6;
ndebriseffects = [-1121474788, -215803600.9, 206173441.7,348961643.4]/1e6;

effects = -1*[ndebriseffects; propellanteffects; powereffects;propsyseffects];

barh(effects, 'stacked')
yticklabels(titles)

% Adjust label font size
set(gca, 'FontSize', 24);
xlabel('\textbf{Effect Size (\$USD million)}');
title('\textbf{Main Effects}', 'FontSize', 26);
legend({'Level 1', 'Level 2', 'Level 3', 'Level 4'}, 'Interpreter', 'latex', 'FontSize', 20);