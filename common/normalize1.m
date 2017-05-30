function r = normalize1(x)

r = (x-min(x(:)))/(max(x(:))-min(x(:)));
