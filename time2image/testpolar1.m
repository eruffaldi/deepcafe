%Encoding Time Series as Images for Visual Inspection and Classification Using Tiled Convolutional Neural...

%%
[p,r] = phirho(cos(1:0.1:50));
polarplot(p,r);

%%
[p,r] = phirho((1:0.1:50).^2);
polarplot(p,r);

%%
g = normalize1(phirhog(cos(1:0.1:50)));
imshow(g)
