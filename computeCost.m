function [J, gradient] = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
%J=sum((X*theta - y).^2)/(2*m);
J= sum(((exp(-(X.*(1/theta(1))).^theta(2)))-y).^2)/(2*m);

gradient1= [((exp(-(X.*(1/theta(1))).^theta(2)))-y)].*[exp(-(X.*(1/theta(1))).^theta(2))].*[X.^theta(2)*theta(2)*(1/theta(1).^(theta(2)+1))].*(1/m);
gradient2= [((exp(-(X.*(1/theta(1))).^theta(2)))-y)].*[exp(-(X.*(1/theta(1))).^theta(2))].*[-((X./theta(1)).^theta(2)).*log(X/theta(1))].*(1/m);

gradient=[sum(sum(gradient1)),sum(sum(gradient2))];
% =========================================================================
size(gradient);
gradient;

end
