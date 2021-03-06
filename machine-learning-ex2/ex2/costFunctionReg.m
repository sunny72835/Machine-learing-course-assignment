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
theta1=theta;
power= (theta' * X')' .* -1;
h= 1 ./ (1 + (e .^ power));
l1=log(h);
l2=log(1-h);
t1= -y'*l1;
t2= -(1-y)'*l2;
t=sum(t1+t2);
J1=(1/m)*t;
theta1(1,:)=[0];
R=((lambda/(2*m))*sum(theta1.^ 2));
J=J1+R;
t=(h-y)' * X;
r=(lambda/m)*theta1;
grad = ((1/m) * t) + r';
grad=grad';




% =============================================================

end
