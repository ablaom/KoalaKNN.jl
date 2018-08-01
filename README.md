# KoalaKNN

A K-nearest neighbor regressor for use in the
[Koala](https://github.com/ablaom/Koala.jl) machine learning
environment.

### Model definition

````
KNNRegressor(K=1, metric=euclidean, kernel=reciprocal)
````

Returns a `K`-nearest neighbor model, with the specified `metric`. A
weighted mean of target values is used in predictions, where the
weight for a neighbor at distance `d` is `kernel(d)`. 

By default numerical features are standardized and categorical
features are one-hot encoded.


### Basic usage

Load libraries and some data:

````julia
julia> using Koala
julia> using KoalaKNN
julia> X, y = load_boston();
````

Instantiate a model:

````julia
julia> knn = KNNRegressor()
KNNRegressor@...569
````

Split rows into training and test:

````julia
julia> train, test = split(eachindex(y), 0.8);
````

Wrap model with data:

````julia
julia> knnM = Machine(knn, X, y, train)
INFO: Element types of input features before transformation:
INFO:   :Crim   => Float64 (continuous)
INFO:   :Zn     => Float64 (continuous)
INFO:   :Indus  => Float64 (continuous)
INFO:   :Chas   => Int64 (categorical)
INFO:   :NOx    => Float64 (continuous)
INFO:   :Rm     => Float64 (continuous)
INFO:   :Age    => Float64 (continuous)
INFO:   :Dis    => Float64 (continuous)
INFO:   :Rad    => Float64 (continuous)
INFO:   :Tax    => Float64 (continuous)
INFO:   :PTRatio        => Float64 (continuous)
INFO:   :Black  => Float64 (continuous)
INFO:   :LStat  => Float64 (continuous)
INFO: 80.0% of data used to compute transformation parameters.
SupervisedMachine{KNNRegressor}@...075
````

Train the resulting supervised learning machine:

````
julia> fit!(knnM, train)
SupervisedMachine{KNNRegressor}@...075
````

Tune the main parameter `K`:

````julia
julia> u, v = @curve K 1:50 begin
       knn.K = K
       fit!(knnM)
       err(knnM, test)
       end;
julia> knn.K = u[indmin(v)]
17
````

Find the cross-validation error for optimal parameter value:

````julia
julia> knn_errs = cv(knnM, eachindex(y))
9-element Array{Float64,1}:
  3.64733
  2.779  
  5.2939 
  6.50681
  5.53358
  3.61784
 10.088  
  4.83911
  3.32053
  
julia> println("error = ", mean(knn_errs), " ± ", std(knn_errs))
error = 5.069568190499833 ± 2.236213912809724
````

To train without standardization of inputs we modify the default
transformer for inputs, build a new machine and retrain:

````julia
julia> t = default_transformer_X(knn)
DataFrameToArrayTransformer@...978

julia> @more
DataFrameToArrayTransformer@...978

key or field            | value
------------------------|------------------------
boxcox                  |false
drop_last               |false
shift                   |true
standardize             |false

julia> t.standardize = false;
julia> knnM = Machine(knn, X, y, train, transformer_X=t);
julia> knn_errs = cv(knnM, eachindex(y))
9-element Array{Float64,1}:
  8.56254
  6.22893
 10.0126 
  8.29487
 12.1207 
  6.40543
 11.3983 
  5.56264
  4.78944

julia> println("error = ", mean(knn_errs), " ± ", std(knn_errs))
error = 8.152833423091781 ± 2.614871487101576
````
