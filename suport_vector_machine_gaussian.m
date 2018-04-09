



%% ................................................
%% ................................................
%% GAUSSIAN SUPORT VECTOR MACHINES
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
load('ex6data3.mat'); 
[m n]= size(X);
fprintf('(X,y) (10 items)\n');   
[X(1:10,:) y(1:10,:)]
fprintf('Program paused. Press enter to continue. \n \n \n');
pause;

if (n==2)
%% 3. Plot training data
fprintf(['Plotting data with + indicating (y = 1) examples and o ' ... 
'indicating (y = 0) examples.\n']);
fprintf('Plot training data ...\n\n'); 
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





%% ======= Part 2: Training SVM with RBF Kernel ===========
fprintf('TRAINING SVM WITH RBF KERNEL\n............................\n \n \n \n');


%% 6. Initial values
%%%%%%********Select C and tol********   
C=0.01;
sigma=0.3;


%% 7. Run svm
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));


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





visualizeBoundary(X, y, model);
fprintf('Program paused. Press enter to continue.\n\n\n');
pause;
endif





%% ============== Part 3: Sample to predict  ==============
fprintf('SAMPLE\n...... \n \n \n \n');





%% 7. Select a sample to predict
%%%%% *************Select sample to predict***********
x11=X(30,:);


%% 8. Estimate the y of the sample
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





[error_train, error_val,ind] = ...
    learningCurve2(X, y, ...
                  Xval, yval, ...
                  C,sigma);


plot(ind, error_train, ind, error_val);
title('Learning curve')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
fprintf('Check if there is a bios or variation problem.\n\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;





%% ================ Part 5: Validation ================
fprintf('\n\nVALIDATION\n.......... \n \n \n \n');









fprintf('Estimation of C and sigma...\n \n \n');
[C1, sigma1] = dataset3Params(X, y, Xval, yval);
fprintf('\n Actual C: \n');
fprintf(' %f \n', C);
fprintf('\n Actual sigma: \n');
fprintf(' %f \n', sigma);
fprintf('\n The best C: \n');
fprintf(' %f \n', C1);
fprintf('\n The best sigma: \n');
fprintf(' %f \n', sigma1);
fprintf('Program paused. Press enter to continue.\n');
pause;














