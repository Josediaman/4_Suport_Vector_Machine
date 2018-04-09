function [error_train, error_val] = ...
    learningCurve(X, y, ...
                  Xval, yval, ...
                  C,tol,max_iter,init_value);
% error_train: error of train set.
% error_val: error of cross validation set.
% X: X train set.
% y: y train set.
% Xval: X cross validation set.
% yval: y cross validation set.
% C: parameter of regularization.
% tol: toleration.
% max_iter: maximun of iterations.
% init_value: initial value of learning curve.



m = size(X, 1);
d=init_value+1;
k=m-d;
error_train = zeros(k, 1);
error_val   = zeros(k, 1);


for i = d:m

	x_train = X(1:i,:);
	y_train = y(1:i);
      model = svmTrain(x_train, y_train, C, @linearKernel, tol, max_iter);
	predictions_train = svmPredict(model, x_train);
	predictions_cross = svmPredict(model, Xval);

	p=i-d+1;
	error_train(p) = mean(double(predictions_train ~= y_train));
	error_val(p) = mean(double(predictions_cross ~= yval));

endfor


end
