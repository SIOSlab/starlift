function [gen,lchrom,coarse,nround] = encode(x,vlb,vub,bits)
%ENCODE Converts from variable to binary representation.
%	[GEN,LCHROM,COARSE,nround] = ENCODE(X,VLB,VUB,BITS) 
%       encodes non-binary variables of X to binary.  The variables 
%       in the i'th column of X will be encoded by BITS(i) bits.  VLB
%       and VUB are the lower and upper bounds on X.  GEN is the binary 
%       representation of these X.  LCHROM=SUM(BITS) is the length of 
%       the binary chromosome.  COARSE(i) is the coarseness of the
%       i'th variable as determined by the variable ranges and 
%       BITS(i).  ROUND contains the absolute indices of the 
%       X which where rounded due to finite BIT length.
%
%	Copyright (c) 1993 by the MathWorks, Inc.
%	Andrew Potvin 1-10-93.

% Remark: what about handling case where length(bits)~=length(vlb)?
lchrom = sum(bits);
coarse = (vub-vlb)./((2.^bits)-1);
[x_row,x_col] = size(x);

gen = [];
if ~isempty(x),
   temp = (x-ones(x_row,1)*vlb)./ ...
          (ones(x_row,1)*coarse);
   b10 = round(temp);
   % Since temp and b10 should contain integers 1e-4 is close enough
   nround = find(b10-temp>1e-4);
   gen = GA_b10to2(b10,bits);
end

% end encode
