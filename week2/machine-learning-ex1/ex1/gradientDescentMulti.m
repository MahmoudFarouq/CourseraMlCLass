function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1);
    thetas = length(theta);
    new_theta = zeros(thetas, 1);

    for iter = 1:num_iters,

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
        for j=1:thetas,
            new_theta(j) = theta(j) - alpha * (1/m) * computeCostDiff(X, y, theta, j); 
        end;
        % ============================================================
        % Save the cost J in every iteration    
        J_history(iter) = computeCost(X, y, theta);
        theta = new_theta;
    end
end

function diff = computeCostDiff(X, y, theta, j)
    m = length(y);
    diff = 0;
    for i=1:m,
      diff += ( theta' * X(i, :)'  - y(i) ) * X(i,j);
    end;
end;
