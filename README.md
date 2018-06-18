# KoalaKNN

A K-nearest neighbor regressor for use in the
[Koala](https://github.com/ablaom/Koala.jl) machine learning
environment.

## Model definition

````
KNNRegressor(K=1, metric=euclidean, kernel=reciprocal)
````

Returns a `K`-nearest neighbor model, with the specified `metric`. A
weighted mean of target values is used in predictions, where the
weight for a neighbor at distance `d` is `kernel(d)`. 

See the documentation for `Koala` for the training API.

