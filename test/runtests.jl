using Koala
using KoalaKNN
using Base.Test
using DataFrames

X = [4 2 5 3;
     2 1 6 0.0]

@test KoalaKNN.distances_and_indices_of_closest(3,
        KoalaKNN.euclidean, X, [1, 1])[2] == [2, 4, 1]

X = DataFrame(X')
y = Float64[2, 1, 3, 8]
knn = KNNRegressor(K=3)
tX = default_transformer_X(knn)
tX.standardize = false
knnM = Machine(knn, X, y, eachindex(y), transformer_X=tX)
fit!(knnM, eachindex(y))

r = 1 + 1/sqrt(5) + 1/sqrt(10)
Xtest = DataFrame([1.0 1.0])
yhat = (1 + 8/sqrt(5) + 2/sqrt(10))/r
@test isapprox(predict(knnM, Xtest)[1], yhat)

X, y = load_boston();
knn = KNNRegressor()
train, test = split(eachindex(y), 0.8);
knnM = Machine(knn, X, y, train)
fit!(knnM, train)

u, v = @curve K 1:50 begin
    knn.K = K
    fit!(knnM)
    err(knnM, test)
end

knn.K = u[indmin(v)]
knn_errs = cv(knnM, eachindex(y))

println("error = ", mean(knn_errs), " ± ", std(knn_errs))

cnst = ConstantRegressor()
cnstM = Machine(cnst, X, y, train)
cnst_errs = cv(cnstM, eachindex(y))

# check KNN is significantly better than constant prediction:
@test compete(cnst_errs, knn_errs, alpha=0.01)[1] == '1'
