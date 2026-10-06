from sklearn.decomposition import PCA

# Defines the covariance matrix
def pca_train(X_train_raw, n) :
    return PCA(n_components = n).fit(X_train_raw)

# Transforms the input data to the k-principal subspace
def pca_transformation(X_raw, pca) :
    return pca.transform(X_raw)
     
# returns the reconstructed data point in the 784-dimensional space
def inverse_pca(X_compressed, pca) : return pca.inverse_transform(X_compressed)