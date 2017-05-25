%Encoding Time Series as Images for Visual Inspection and Classification Using Tiled Convolutional Neural...

%%
[p,r] = phirho(cos(1:0.1:50));
polarplot(p,r);
%%

g = normalize1(phirhog(cos(1:0.1:50)));
imshow(g)

%%
[p,r] = phirho((1:0.1:50).^2);
polarplot(p,r);

%%

[g,s] = phirhog((1:0.1:50).^2);
g = normalize1(g);
s = normalize1(s);
figure(1)
imshow(g)
figure(2)
imshow(s)
%%

x = 1:0.1:50;
g = normalize1(phirhog(log(1./x).*cos(x)));
imshow(g)

%%
load gas
g = normalize1(phirhog(price1));
imshow(g)
