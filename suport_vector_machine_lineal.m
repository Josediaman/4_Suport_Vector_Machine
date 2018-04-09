 




%% ................................................
%% ................................................
%% LINEAL SUPORT VECTOR MACHINES
%% ................................................
%% ................................................





%% 1. Clear and Close Figures
clear ; close all; clc





%% ==================== Part 1: Data ====================
fprintf('\n \nDATA\n.... \n \n \n');   





%% 2. Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add your own file

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fprintf('Loading data ...\n'); 
%%%%%%********Select archive********   
load('ex6data1.mat'); 
[m n]= size(X);
fprintf('(X,y) (10 items)\n');   
[X(1:10,:) y(1:10,:)]
fprintf('Program paused. Press enter to continue. \n \n \n');
pause;

if (n==2)
%% 3. Plotting Data
fprintf(['Plotting data with + indicating (y = 1) examples and o ' ... 
'indicating (y = 0) examples.\n']);
plotData(X, y);
fprintf('Program paused. Press enter to continue.\n \n \n \n');
pause;
endif

%% 4. Normalizing Features and adding first colum of ones
fprintf('Normalizing Features and adding first colum of ones ...\n');
[m, n] = size(X); 
[X mu sigma] = featureNormalize(X);
fprintf('X (normal) (10 items)\n');
X(1:10,1:n)
fprintf('Program paused. Press enter to continue.\n \n \n \n');
pause;


%% 5. Select train, cross and test validation sets
[X, y, Xval, yval, Xerr, yerr, m, n] = ...
    selectsets(X, y);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% extract sets

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%% =========== Part 2: Training Linear SVM ===============
fprintf('TRAINING LINEAR SVM\n....................\n \n \n \n');




%% 6. Initial values
%%%%%%********Select C and tol********   
C = 1;
tol = 1e-3;
max_iter = 20;


%% 7. Run svm 
model = svmTrain(X, y, C, @linearKernel, tol, max_iter);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Optional: Execute your own gradient descent.

%fprintf('Running gradient descent with alpha ... \n \n ');
%%%%% *************Select iterations***********
%num_iters = 1000;
%[theta, J_his] = gradientDescentMulti(X, y, theta, alpha, %num_iters);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


p = svmPredict(model, X);
p2 = svmPredict(model, Xval);
p3 = svmPredict(model, Xerr);

error = mean(double(p ~= y));
error2 = mean(double(p2 ~= yval));
error3 = mean(double(p3 ~= yerr));

%% 8. Display results
fprintf('Train Accuracy: %f \n', mean(double(p == y)) * 100);
fprintf('Cross Accuracy: %f \n', mean(double(p2 == yval)) * 100);
fprintf('Test Accuracy: %f \n\n\n\n', mean(double(p3 == yerr)) * 100);
fprintf('Train Error: %f \n', error);
fprintf('Cross Error: %f \n', error2);
fprintf('Test Error: %f \n', error3);
fprintf('Program paused. Press enter to continue.\n\n\n');
pause;




if (n==2)
%% ================ Part 2': GRAPHIC ================
fprintf('GRAPHIC \n...... \n \n \n \n');






visualizeBoundaryLinear(X, y, model);
fprintf('Program paused. Press enter to continue.\n\n\n');
pause;
endif





%% ============== Part 3: Sample to predict  ==============
fprintf('SAMPLE\n...... \n \n \n \n');





%% 7. Select a sample to predict
%%%%% *************Select sample to predict***********
x11=X(30,:);


%% 11. Estimate the y of the sample
y_estimation=svmPredict(model,x11);
fprintf('Prediction:\n x= \n');
fprintf('%f  \n',x11(1, :));
fprintf('...\n');
fprintf('\n y_pred= %f',y_estimation);
fprintf('\n y_real= %f \n \n',y(30));
fprintf('Program paused. Press enter to continue.\n');
pause;





%% ==== Part 5: Learning Curve ========
fprintf('\n\n LEARNING CURVE\n............... \n \n \n \n');





[error_train, error_val] = ...
    learningCurve(X, y, ...
                  Xval, yval, ...
                  C,tol,max_iter,2);

d=3;
plot(d:m, error_train, d:m, error_val);
title('Learning curve')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
fprintf('Check if there is a bios or variation problem.\n\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;





%% ================ Part 5: Validation ================
fprintf('\n\nVALIDATION\n.......... \n \n \n \n');





[C_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval, tol,max_iter);

figure;
plot(C_vec, error_train, C_vec, error_val);
legend('Error Train', 'Error Cross');
xlabel('C');
ylabel('Error');
fprintf('C\t\t\tTrain Error\tValidation Error\n');
for i = 1:length(C_vec)
	fprintf(' %f\t%f\t%f\n', ...
            C_vec(i), error_train(i), error_val(i));
end


fprintf('\n Actual C: \n');
fprintf(' %f \n', C);
fprintf('\nThe best C has the lowest validation error.\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;







