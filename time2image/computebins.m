% Compute Bins
%
% Input:
%   X data
%   n number of (odd) bins (default 3)
%   dim dimension along with (default 2)
%
% Output:
%   Q are the center of percentiles
%   p are the percentiles of Q
%   QE are the bins edges (for discrete)
%
% Use discrete(data,QE) for discretizing the input
%
% Emanuele Ruffaldi
function [Q,p,QE] = computebins(X,n,dim)
if nargin < 2
    n = 3;
end
if nargin < 3
    dim = 2;
end
assert(n >= 1,'required at least one');
assert(mod(n,2) == 1,'Required odd bins');

p = 50;
for I=3:2:n
    p = [p(1)-p(1)/2 p p(end)+p(end)/2];
end

Q = prctile(X,p,dim);

QE = interp1(1:length(Q),Q,(1:length(Q)-1)+0.5,'linear');
Q
QE

QE = [min(X,[],dim) QE max(X,[],dim)];