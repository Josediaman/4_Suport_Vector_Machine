function plotData(X, y)
% X: train examples.
% y: values of labels (y=1 or y=0).


pos = find(y == 1); 
neg = find(y == 0);

plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 1, 'MarkerSize', 7)
hold on;
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)
hold off;



end
