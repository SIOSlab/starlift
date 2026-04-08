function [new_gen,sites] = xover(old_gen,Pc)
%XOVER  Creates a NEW_GEN from OLD_GEN using crossover.
%	[NEW_GEN,SITES] = XOVER(OLD_GEN,Pc) performs crossover
%       procreation on pairs of OLD_GEN with probability Pc.
%       Crossover SITES are chosen at random (re: there will be
%       half as many SITES as there are individuals.
%
%	Copyright (c) 1993 by the MathWorks, Inc.
%	Andrew Potvin 1-10-93.

lchrom = size(old_gen,2);
sites = ceil(rand(size(old_gen,1)/2,1)*(lchrom-1));
sites = sites.*(rand(size(sites))<Pc);
for i = 1:length(sites);
   new_gen([2*i-1 2*i],:) = [old_gen([2*i-1 2*i],1:sites(i)) ...
                             old_gen([2*i 2*i-1],sites(i)+1:lchrom)];
end 

% end xover
