
a = (normrnd(0,5,100,1)');
a = cos(1:0.2:100);
[Q,p,QE]= computebins(a,3);
ad = discretize(a,[min(a),Q,max(a)]);

subplot(2,1,1);
plot(QE(ad))
hold on;
xl = xlim;
for I=1:length(Q)
    line(xl,[Q(I),Q(I)]);
end
plot(a,'--r');
hold off
subplot(2,1,2);
hist(ad)

% build matrix
% wij = 1..n
%
% normalize rows