function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



sum = 0;
for i=1:m,
  sum += ( theta' * X(i, :)'  - y(i) )^2;
end;
sum2 = 0;
for j=2:size(theta, 1),
  sum2 += theta(j)^2;
end;
J = (1/(2*m)) * sum +  (lambda/(2*m)) * sum2;




for j=1:rows(grad),
  s = 0;
  for i=1:m,
    s += ( theta' * X(i, :)'  - y(i) ) * X(i, j);
    if j != 1,
      s += lambda/m * theta(j);
    end;
  end;
  grad(j) = 1/m * ( s );
end;





% =========================================================================

grad = grad(:);

end
