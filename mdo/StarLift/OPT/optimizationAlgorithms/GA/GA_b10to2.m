function b2 = b10to2(b10,bits)
%B10TO2 Converts base 10 to base 2.
%       X = B10TI2(N,BITS) returns a vector of size BITS of the binary
%       representation of the base 10 integer N.  If N is a matrix,
%       BITS must be a row vector with as many columns as N.  X will
%       then be of size (N,1)xSUM(BITS).
%
%       Copyright (c) 1993 by The MathWorks, Inc.
%       Andrew Potvin 1-10-93

bit_count = 0;
b2_index = [];
bits_index = 1:length(bits);
for i=bits_index,
   bit_count = bit_count + bits(i);
   b2_index = [b2_index bit_count];
end

for i=1:max(bits),
   r = rem(b10,2);
   b2(:,b2_index) = r;

   b10 = fix(b10/2);
   tbe = find( all(b10==0) | (bits(bits_index)==i) );
   if ~isempty(tbe),
      b10(:,tbe) = [];
      b2_index(tbe) = [];
      bits_index(tbe) = [];
   end

   % Quick quit if all b10 small compared to bit length
   if isempty(bits_index),
      return
   end
   
   b2_index = b2_index-1;

end

% end
