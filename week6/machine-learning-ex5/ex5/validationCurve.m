function [lambda_vec, error_train, error_val] = validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(size(lambda_vec));
error_val   = zeros(size(lambda_vec));

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%



for i=1:length(lambda_vec),
  theta = trainLinearReg(X, y, lambda_vec(i));
  error_train(i) = _linearRegCostFunction(X,    y,    theta, lambda_vec(i));
  error_val(i)   = _linearRegCostFunction(Xval, yval, theta, lambda_vec(i));
end;

% =========================================================================

end


function J = _linearRegCostFunction(X, y, theta, lambda)
  m = length(y);
  sum = 0;
  for i=1:m,
    sum += ( theta' * X(i, :)'  - y(i) )^2;
  end;
  sum2 = 0;
  for j=2:size(theta, 1),
    sum2 += theta(j)^2;
  end;
  J = (1/(2*m)) * sum +  (lambda/(2*m)) * sum2;
end;








