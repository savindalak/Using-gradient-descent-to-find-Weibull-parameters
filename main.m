%% Gradient descent application to find the parameters of Weibull
%% Reliability function for data set given in reliability_data.m

% X refers to the time in hours
% y refers to the Reliability
%

%% Initialization
clear ; close all; clc

%% ======================= Plotting the data================
fprintf('Plotting Data ...\n')
data = load('reliability_data.txt');
X = data(:, 1); y = (data(:, 2));
m = length(y); % number of training examples

plotData(X, y);
xlabel('time'); ylabel('Reliability');
title('Reliability vs time plot');
fprintf('Program paused. Press enter to continue.\n');
kbhit();

%% =================== compute Cost and Gradient descent ===================

theta = [875;1.25]; % initialize fitting parameters for neta and beta of reliability function

% Some gradient descent settings
iterations = 2500;
alpha = 50;

fprintf('\nTesting the cost function ...\n')
% compute and display initial cost
J = computeCost(X, y, theta);
fprintf('With theta = [875 ; 1.24]\nCost computed = %f\n', J);


fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
[theta,J_cost] = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('neta and beta of Weibull Reliability function found by gradient descent:\n');
fprintf('%f\n', theta);
%fprintf('%f\n', J_cost);

% Plot the linear fit
hold on; % keep previous plot visible
fprintf('plotting the graph with found neta and beta \n');
fprintf('Program paused. Press enter to continue.\n');
kbhit();
plot(X(:), (exp(-(X.*(1/theta(1))).^theta(2))), '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure


% Predict values for Reliability at 500 hr and 900 hr
predict1 = (exp(-(500.*(1/theta(1))).^theta(2)));
fprintf('At 500hrs, we predict a Reliability of %f\n',...
    predict1);
predict2 = (exp(-(900.*(1/theta(1))).^theta(2)));
fprintf('At 900hrs, we predict a Reliability of %f\n',...
    predict2);

fprintf('Program paused. Press enter to continue.\n');
kbhit();

%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(50, 2000, 10);
theta1_vals = linspace(0, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('neta'); ylabel('\beta');
title('Surface plot of cost function');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('neta'); ylabel('beta');
title('Contour plot of cost function');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

%% ======================= verify results with adavanced optimisation algorithms================
options = optimset('GradObj', 'on', 'MaxIter', 2500);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
initial_theta = [875,1.25];
[theta, cost] = ...
	fminunc(@(t)(computeCost(X, y,t)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

 %%==================== make predictions======================
 
 X = 1300;
 prediction = (exp(-(X.*(1/theta(1))).^theta(2)));
 fprintf(['Predicted reliability at 1300 hrs (using our Weibull function):\n %f\n'], prediction);
