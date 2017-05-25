function g = phirhog(x)

x = normalize1(x);

K = sqrt(ones(length(x))-x'*x);
g = x'*x-K'*K;