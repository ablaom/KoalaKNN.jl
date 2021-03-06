module KoalaKNN

# new:
export KNNRegressor 

# needed in this module:
import Koala: Regressor, softwarn, SupervisedMachine, params
import KoalaTransforms: DataFrameToArrayTransformer
import DataFrames: AbstractDataFrame # the form of all untransformed input data

# to be extended (but not explicitly rexported):
import Koala: setup, fit, predict
import Koala: default_transformer_X, default_transformer_y, clean!
import Koala: transform, inverse_transform


## MODEL TYPE DEFINITIONS

KNNPredictorType = Tuple{Matrix{Float64},Vector{Float64},DateTime}

mutable struct KNNRegressor <: Regressor{KNNPredictorType}
    K::Int           # number of local target values averaged
    metric::Function
    kernel::Function # each target value is weighted by `kernel(distance^2)`
end

euclidean(v1, v2) = norm(v2 - v1)
reciprocal(d) = 1/d

# lazy keywork constructor:
function KNNRegressor(; K=1, metric=euclidean, kernel=reciprocal)
    model = KNNRegressor(K, metric, kernel)
    softwarn(clean!(model)) # only issues warning if `clean!` changes `model`
    return model
end

function clean!(model::KNNRegressor)
    message = ""
    if model.K <= 0
        model.K = 1
        message = message*"K cannot be negative. K set to 1."
    end
    return message
end

function Base.showall(stream::IO, mach::SupervisedMachine{KNNPredictorType, KNNRegressor})
    dict = params(mach)
    report_items = sort(collect(keys(dict[:report])))
    dict[:report] = "Dict with keys: $report_items"
    dict[:Xt] = string(typeof(mach.Xt), " of shape ", size(mach.Xt))
    dict[:yt] = string(typeof(mach.yt), " of shape ", size(mach.yt))
    delete!(dict, :cache)
    dict[:predictor] = "(based directly on data)"
    showall(stream, mach, dict=dict)
    println(stream, "\n## Model detail:")
    showall(stream, mach.model)
end


## TRANSFORMERS

default_transformer_X(model::KNNRegressor) =
    DataFrameToArrayTransformer(standardize=true)


## TRAINING AND PREDICTION METHODS

# computing norms of rows later on is faster if we use the transpose of X
setup(model::KNNRegressor, X, y, scheme_X, parallel, verbosity) = (X', y)

function fit(model::KNNRegressor, cache, add, parallel, verbosity)
    return ((cache..., now()), Dict{Symbol,Any}(), cache)
end

first_component_is_less_than(v, w) = isless(v[1], w[1])

function distances_and_indices_of_closest(K, metric, Xtrain, pattern)

    distance_index_pairs = Array{Tuple{Float64,Int}}(size(Xtrain, 2))
    for j in 1:size(Xtrain, 2)
        distance_index_pairs[j] = (metric(Xtrain[:,j], pattern), j)
    end

    sort!(distance_index_pairs, lt=first_component_is_less_than)
    distances = Array{Float64}(K)
    indices = Array{Int}(K)
    for j in 1:K
        distances[j] = distance_index_pairs[j][1]
        indices[j] = distance_index_pairs[j][2]
    end

    return distances, indices    
    
end

function predict_on_pattern(model, predictor, pattern)
    Xtrain, ytrain = predictor[1], predictor[2]
    distances, indices = distances_and_indices_of_closest(model.K, model.metric, Xtrain, pattern)
    wts = [model.kernel(d) for d in distances]
    wts = wts/sum(wts)
    return sum(wts .* ytrain[indices])
end

predict(model::KNNRegressor, predictor::KNNPredictorType,
        X, parallel, verbosity) = [predict_on_pattern(model, predictor, X[i,:])
                                   for i in 1:size(X,1)]


end # of module


