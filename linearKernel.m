function sim = linearKernel(x1, x2)
% sim: Linear Kernel.
% x1: variable 1.
% x2: variable 2.



x1 = x1(:); x2 = x2(:);
sim = x1' * x2;  % dot product




end