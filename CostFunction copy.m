function [J, grad] = CostFunction(params, Y, R, num_users, num_jokes, num_features, lambda)

X = reshape(params(1:num_jokes*num_features), num_jokes, num_features);
Theta = reshape(params(num_jokes*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

%===================================================

err = X * Theta' - Y;
err = err .* R;

J = 0.5 * sumsq(err(:));

X_grad = err * Theta;

Theta_grad = err' * X;

% regularization

J = J + (lambda / 2) * (sumsq(Theta(:)) + sumsq(X(:)));

X_grad = X_grad + lambda * X;

Theta_grad = Theta_grad + lambda * Theta;

%=======================================

grad = [X_grad(:); Theta_grad(:)];

end
