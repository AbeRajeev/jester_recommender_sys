clear ; close all; clc

% Y = 100 x 1000; # 100 jokes and 1000 users
% R = 100 x 1000; # did rate matrix
% the data is classified in a way that 100 jokes are rated by 1000 users.
% the file ratings.mat contains both Y and R. 'ratings.mat' is generated in the Matlab cmd window.
% The code to generate 'ratings.mat' are 
% Y = jestermini' # the file which we had earlier.
% R = Y ~= 0; # after this save both the files using the save command. 
% save('ratings','Y','R');

fprintf('load the ratings and the matrix which shows whether the user has rated')

load('ratings.mat');
% we can do some printing and stuff in here.
imagesc(Y);
xlabel('Users');
ylabel('Jokes');

% providing our own rating to the jokes
my_ratings = zeros(100, 1);

my_ratings(1) = 4.64;
my_ratings(98) = -2.43;
my_ratings(7) = 6.56;
my_ratings(12)= 5.23;
my_ratings(54) = -4.65;
my_ratings(64)= -5.75;
my_ratings(66)= 8.67;
my_ratings(69) = 9.45;
my_ratings(18) = 4.34;
my_ratings(22) = -5.65;
my_ratings(35)= 5.97;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%adding our ratings to the matrix

%load the data again, i am not exactly sure why are we loading again. Maybe because the first time we have used to print.

load('ratings.mat');

Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

% values which we are gonna use
num_users = size(Y, 2);
num_jokes = size(Y, 1);
num_features = 5;

% Initialize Parameters Theta (user_prefs), X (features)
features = randn(num_jokes, num_features);
user_prefs = randn(num_users, num_features);
initial_parameters = [user_prefs(:); features(:)];

% set options to perform the gradient optimization 
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 10;
theta = fmincg (@(t)(CostFunction(t, Ynorm, R, num_users, num_jokes, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% unfolding the X and Theta values 				
X = reshape(theta(1:num_jokes*num_features), num_jokes, num_features);
Theta = reshape(theta(num_jokes*num_features+1:end), ...
                num_users, num_features);
				
				
fprintf('Recommender system learning completed.\n');			

p = X * Theta';
my_predictions = p(:,1) + Ymean;

%my_predictions % this displays the predicted ratings

%save('weights','features','user_prefs'); %# here the learned features and the parameters are saved into a new file.

%==================================== Recommendation system compoleted ===========================

%==== no on we can use k means clustering to cluster the data(ratings) based on the features that we have obtained.

load('weights.mat'); % it is filled with the features that the algorithm has calculated, X is the predicted features.
%load('ratings.mat'); % above action is more preferred as the simlar jokes will given out to the users. This is ...
						% ...good too but only when the user goes behind the ratings instead of features.

K = 5;
max_iters = 10;

initial_centroids = kMeansInitCentroids(X, K);

idx = findClosestCentroids(X, initial_centroids);
centroids = computeCentroids(X, idx, K);

[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);

fprintf('k means clustering successfully performed\n')



