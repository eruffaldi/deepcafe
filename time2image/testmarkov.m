
a = (normrnd(0,5,100,1)');
a = cos((1:0.2:100).^2);
[Q,QE]= computebins(a,5); % TODO fixme
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
hist(ad,1:max(ad))

% build matrix
% wij = 1..n
%
% normalize rows