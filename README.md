# jester_recommender_sys

Recommendation system for the jester (joke) database using collaborative filtering and K-means
clustering algorithms. 

Author: Abhijith Rajeev (Abe). Project date: Jan 2017. Libraries and Code references: Introduction to Machine Learning - Andrew Ng.

************ Project Overview ******************

● The jester database has the ratings of 100 jokes from 73,421 users. Link to the datset- (http://eigentaste.berkeley.edu/dataset/) .
● Ratings of 1000 users are considered for convenience. New user ratings are appended to the data.
● Parameter learning and feature learning is performed using advanced optimization algorithms (ex: fmincg).
● Polynomial regression is performed (as a part of collaborative filtering algorithm) to predict the ratings.
● All the jokes with similar features using the K Nearest Neighbors - K means Clustering algorithm, so that the jokes can be taken from certain groups to give to the users.
