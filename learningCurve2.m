function [error_train, error_val,ind] = ...
    learningCurve2(X, y, ...
                  Xval, yval, ...
                  C,sigma);
% error_train: error of train set.
% error_val: error of cross validation set.
% ind: index of number of examples.
% X: X train set.
% y: y train set.
% Xval: X cross validation set.
% yval: y cross validation set.
% C: parameter of regularization.
% sigma: parameter of regulation.




m = size(X, 1);
num=10;
error_train = zeros(num+1, 1);
error_val   = zeros(num+1, 1);
valu=floor(m/num)-5;
ind=0;




for i=1:num+1,

	value=(i-1)*valu+5;
	ind=[ind value];
     fprintf('\n Case m= %f\n',ind(i+1));
	x_train = X(1:ind(i+1),:);
	y_train = y(1:ind(i+1));

      model= svmTrain(x_train, y_train, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

	predictions_train = svmPredict(model, x_train);
	predictions_cross = svmPredict(model, Xval);
	error_train(i) = mean(double(predictions_train ~= y_train));
	error_val(i) = mean(double(predictions_cross ~= yval));

endfor


ind=ind(2:end);


end




