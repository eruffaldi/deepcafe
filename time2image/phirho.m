function [phi,ro] = phiro(x)
% normalized -1,1
x = normalize1(x);
phi = acos(x);
ro = (0:length(x)-1)/length(x);






