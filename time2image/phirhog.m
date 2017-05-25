function [g,s] = phirhog(x)

x = normalize1(x);

K = sqrt(ones(length(x))-x'*x);
g = x'*x-K'*K;
if nargout > 1
    s = K'*x'-x*K;
end
    