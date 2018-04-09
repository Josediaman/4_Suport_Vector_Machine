function [C, sigma] = dataset3Params(X, y, Xval, yval)
% C: parameter of regularization.
% sigma: parameter of regulation.
% X: X train set.
% y: y train set.
% Xval: X cross validation set.
% yval: y cross validation set.


c_row = [0.01 0.03 0.1 0.3 1 3 10 30];
sig_row = [0.01 0.03 0.1 0.3 1 3 10 30];
error = zeros(size(c_row,2),size(sig_row,2));


for i = 1:size(c_row,2),

	c_par = c_row(i);
	for j = 1:size(sig_row,2),
	sig = sig_row(j);
	model = svmTrain(X, y, c_par, @(x1, x2) 	gaussianKernel(x1, x2, sig)); 
	predictions = svmPredict(model, Xval);
	error(i,j) = mean(double(predictions ~= yval))
	endfor

endfor


[val c_pos] = min(min(error'));
[val sigma_pos] = min(min(error));


C = c_row(c_pos);
sigma = sig_row(sigma_pos);


end



