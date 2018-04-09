function [C_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval, tol,max_iter);

% C_vec: values of lambda.
% error_train: error of train set.
% error_val: error of cross validation set.
% X: X train set.
% y: y train set.
% Xval: X cross validation set.
% yval: y cross validation set.
% tol: toleration.
% max_iter: maximum number of iterations.


         
C_vec = [0 0.001 0.003 0.01 0.03 0.1 0.5 1 1.5 2]';
error_train = zeros(length(C_vec), 1);
error_val = zeros(length(C_vec), 1);

for i = 1:length(C_vec)
         
	C = C_vec(i,1);
	model = svmTrain(X, y, C, @linearKernel, tol, max_iter);
	predictions_train = svmPredict(model, X);
	predictions_cross = svmPredict(model, Xval);

	error_train(i) = mean(double(predictions_train ~= y));
	error_val(i) = mean(double(predictions_cross ~= yval));


end


end
