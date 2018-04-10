function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


sum = 0;
for i=1:m,
  sum += -y(i) * log( sigmoid( theta' * X(i, :)' ) ) - (1 - y(i) ) * log( 1 - sigmoid( theta' * X(i, :)' ) );
end;
sum2 = 0;
for j=2:size(theta, 1),
  sum2 += theta(j)^2;
end;
J = (1/m) * sum +  (lambda/(2*m)) * sum2;




for j=1:rows(grad),
  s = 0;
  for i=1:m,
    s += ( sigmoid( theta' * X(i, :)' ) - y(i) ) * X(i, j);
    if j != 1,
      s += lambda/m * theta(j);
    end;
  end;
  grad(j) = 1/m * ( s );
end;



% =============================================================

end
