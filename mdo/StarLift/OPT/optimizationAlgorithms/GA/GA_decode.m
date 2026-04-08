function [x,coarse] = decode(gen,vlb,vub,bits)
%DECODE Converts from binary to variable representation.
%	[X,COARSE] = DECODE(GEN,VLB,VUB,BITS) converts the binary 
%       population GEN to variable representation.  Each individual 
%       of GEN should have SUM(BITS).  Each individual binary string
%       encodes LENGTH(VLB)=LENGTH(VUB)=LENGTH(BITS) variables.
%       COARSE is the coarseness of the binary mapping and is also
%       of length LENGTH(VUB).
%
%	Copyright (c) 1993 by the MathWorks, Inc.
%	Andrew Potvin 1-10-93.

bit_count = 0;
two_pow = 2.^(0:max(bits))';
for i=1:length(bits),
   pow_mat((1:bits(i))+bit_count,i) = two_pow(bits(i):-1:1);
   bit_count = bit_count + bits(i);
end

gen_row = size(gen,1);
coarse = (vub-vlb)./((2.^bits)-1);
inc = ones(gen_row,1)*coarse;
x = ones(gen_row,1)*vlb + (gen*pow_mat).*inc;

% end decode
