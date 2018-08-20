Recommendation system for the jester (joke) database using collaborative filtering and K-means clustering algorithms.

- Jester data set includes the ratings of 100 jokes from 73421 users. Rating is in the range of -10 and 10.
- Here only part I of the dataset is considered, which has ratings of 24,983 users. 

Design considerations:
- Rating 99 means that the joke has not been rated (null).
- Eigenstate clusters all the jokes of similar ratings and gives to the new users according to their humor level.

Steps:
- Users gives ratings to few jokes and the system learns the pattern and rates rest of the jokes according to the given user ratings. 
- Similarly to the given recommendation system, it gives the predicted ratings.

Details:
- In the command 'imagesc()', 'Y axis' is users and 'x axis' is number of jokes.
- 'R matrix' is created with the values 1s and 0s corresponding to ratings given or not.

- Now the parameter 'theta' and the feature 'x' need to be learned;
	- Collobarative filtering allows the feature learning.
	- And parameter learning is achieved by using an advanced learning algorithm such as 'fmincg'.
	
	- Guess (random generate)
		theta -> x (calculated) -> theta -> x

	- collobarative filtering is more efficient so there is no need going back and forth from x to theta.

Note: There is no need to set the bias term as 1, the algorithm is good enough to learn that.

	- With the feature 'x' calculated, we can perform 'pattern recognition' to find the related jokes.


Data: 
jester: 1000 x 100
R: 1000 x 100
theta: 100 x 10 (weights)
x: 1000 x 10 (features)

'ratings.mat'
Y = 100 x 1000; 100 jokes, 1000 users
R = 100 x 1000; ratings matrix - whether its been rated or not
'myratings' = 100, 1

-----------------------------------------------------

- Once the Recommendation system learning is complete, the learned parameters can be saved using the command that has been commented in the end of the file. 
- Next K means clustering is implemented to group the ratings of similar interest.
- These grouped ratings can be given to the new user according to their interests. 

- K means clustering is performed on the features matrix 'x'.









