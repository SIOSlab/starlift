function [] = sensMultiPlot(sensChart,labels)
%sensMultiPlot Takes in a big sensitivity plot and divides it into smaller
%plots for legibility

barsPerChart = 5;

set(groot,'defaultAxesTickLabelInterpreter','latex')
numPlots = ceil(length(sensChart)/barsPerChart); %how many individual plots do I need to put n charts on each 

[~,I] = sort(abs(sensChart(:,1))+abs(sensChart(:,2)));
neg = sensChart(:,1);
pos = sensChart(:,2);
negSorted = neg(I);
posSorted = pos(I);

newChart = [negSorted posSorted];

labelSorted = labels(I);


for i = 1:numPlots
startIndex = barsPerChart*(i-1)+1;
endIndex = i*barsPerChart;
figure

if endIndex > length(sensChart)
    endIndex = length(sensChart);
end
x = categorical(labelSorted(startIndex:endIndex));
x = reordercats(x,labelSorted(startIndex:endIndex));

barh(x,newChart(startIndex:endIndex,:), 'stacked', 'BaseValue',0)
titletext = sprintf("\\bf{Perturbations %d through %d}",startIndex, endIndex);
title(titletext, 'Interpreter','latex')
xlabel('\bf{Normalized sensitivities}','Interpreter','latex')
ylabel('\bf{Parameters}','Interpreter','latex')
yticklabels(labelSorted(startIndex:endIndex))
% set(gca,'yticklabel',labels)
legend({'Negative step','Positive step'})
end

end

