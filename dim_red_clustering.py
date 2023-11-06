from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from sklearn.cluster import Birch
import pandas as pd
from scipy.io import arff
import preprocessing
import kmeans
import postprocessing
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------Dimensionality Reduction Function-----------------------------------------------

def dim_red_clustering(X,labels_true, dim_red=None, dim_red_params=None, alg ='kmeans', alg_params = None):
    """
    Its main objective is to encapsulate the dimensionality reduction process so it can be called repeatedly from main
    This function receives as input:
        - X (nd-array): feature dataset
        - y (array): true labels
        - dim_red (str): {pca, incr_pca, trunc_svd, pca_mai} / None
        - dim_red_params (dict): list of parameters for the dim. red. algorithm / None
        - alg (str): {kmeans, birch}
        - alg_params (dict): list of parameters for the clustering algorithm
    Outputs: Dataset without classes transformed
    """

    # Compute algorithms without dim. red. is defined
    if dim_red is None:
        if alg_params is None:
            dim_red_params = {}
        if alg == 'kmeans':
            labels_pred, C_store_best = kmeans.kmeans(X, **alg_params)
        if alg == 'birch':
            labels_pred = Birch(**alg_params).fit_predict(X)

    # Compute algorithms with dim_red_tech
    else:
        # Dimensionality reduction
        if dim_red_params is None:
            dim_red_params = {}
        if dim_red == 'pca':
            dim_red = PCA(**dim_red_params)
            X_transformed = dim_red.fit_transform(X)
        if dim_red == 'incr_pca':
            dim_red = IncrementalPCA(**dim_red_params)
            X_transformed = dim_red.fit_transform(X)
        if dim_red == 'trunc_svd':
            dim_red = TruncatedSVD(**dim_red_params)
            X_transformed = dim_red.fit_transform(X)
        if dim_red == 'pca_mai':
            # implement
            '''function that levi will implement'''

        # Algorithm on X_transformed
        if alg_params is None:
            dim_red_params = {}
        if alg == 'kmeans':
            labels_pred, C_store_best = kmeans.kmeans(X_transformed,**alg_params)
        if alg == 'birch':
            labels_pred = Birch(**alg_params).fit_predict(X_transformed)

    return X_transformed, labels_pred

