function [new_gen,mutated] = mutate(old_gen,Pm)
%MUTATE Changes a gene of the OLD_GEN with probability Pm.
%	[NEW_GEN,MUTATED] = MUTATE(OLD_GEN,Pm) performs random
%       mutation on the population OLD_POP.  Each gene of each
%       individual of the population can mutate independently
%       with probability Pm.  Genes are assumed possess boolean
%       alleles.  MUTATED contains the indices of the mutated genes.
%
%	Copyright (c) 1993 by the MathWorks, Inc.
%	Andrew Potvin 1-10-93.

mutated = find(rand(size(old_gen))<Pm);
new_gen = old_gen;
new_gen(mutated) = 1-old_gen(mutated);

% end mutate
