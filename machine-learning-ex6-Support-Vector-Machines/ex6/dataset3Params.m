function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = .1;
x1 = [1 2 1]; x2 = [0 4 -1]; 
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vect =[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vect =[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30,32];

for i2 =1:length(C_vect)
    for j2 =1:length(sigma_vect)
        model= svmTrain(X, y, C_vect(i2), @(x1, x2) gaussianKernel(x1, x2, sigma_vect(j2)));
        pred = svmPredict(model,Xval);
        accuracy_matrix(i2,j2) = mean(double(pred ~= yval));
    end
end




% =========================================================================

end
