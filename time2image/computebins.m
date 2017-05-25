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
function [Q,QE] = computebins(X,n,dim)
if nargin < 2
    n = 3;
end
if nargin < 3
    dim = 2;
end
assert(n >= 1,'required at least one');
assert(mod(n,2) == 1,'Required odd bins');

Q = quantile(X,n);
QE = interp1(1:length(Q),Q,(1:length(Q)-1)+0.5,'linear');

QE = [min(X,[],dim) QE max(X,[],dim)];