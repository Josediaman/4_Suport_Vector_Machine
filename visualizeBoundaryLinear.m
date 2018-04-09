function visualizeBoundaryLinear(X, y, model)
% X: Training examples.
% y: Training predictions.
% model: Parameters of the model.



w = model.w;
b = model.b;
xp = linspace(min(X(:,1)), max(X(:,1)), 100);
yp = - (w(1)*xp + b)/w(2);
plotData(X, y);
hold on;
plot(xp, yp, '-b'); 
hold off

end
