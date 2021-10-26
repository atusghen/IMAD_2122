%% Load Data ex 2: house prices estimation 

clc
clear
close all

data = load('Data\ex1data2.txt');
X = data(:, 1:2); % regressors matrix
y = data(:, 3); % output
N = length(y); % number of training samples \ observations

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');


%% Least squares - normal equations

% ====================== YOUR CODE HERE ===================================
% Instructions: Compute the estimate for linear regression using normal equations
X = [ones(N, 1), X];
% You need to estimate the following variables correctly 
theta_hat = pinv(X'*X)*X'*y; % insert here the code, instead of " = 0";
X = data(:, 1:2);

% =========================================================================


% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta_hat);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
price_hat = [1 1650 3]*theta_hat; 

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price_hat);
fprintf('\n');
fprintf('\n');


%% Gradient Descent

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[Z mu sigma] = featureNormalize(X); % normalize data

% Add intercept term to normalized data matrix Z
Xaug = [ones(N, 1) Z];
fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 500;

% Init Theta and Run Gradient Descent 
theta_hat = zeros(3, 1);
[theta_hat, J_history] = gradientDescentMulti(Xaug, y, theta_hat, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J'); grid on;

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta_hat);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.

z = ([1650 3] - mu)./sigma; % now you should work with normalized data
z = [1 z]; % add dummy regressor for intercept term to normalized data
price_hat = z*theta_hat ;  % estimated price

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price_hat);
 