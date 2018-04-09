function sim = gaussianKernel(x1, x2, sigma)
% sim: gaussian Kernel.
% x1: variable one.
% x2: variable two.
% sigma: parameter of the kernel.


x1 = x1(:); x2 = x2(:);
sim = 0;
vect = x1-x2;
sim = exp(-(vect'*vect)/(2*(sigma^2)));

    
end
