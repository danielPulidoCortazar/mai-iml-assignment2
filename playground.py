import pandas as pd
from scipy.io import arff
from preprocessing import preprocessing_pipeline, binClass, decodeBytesAndReplaceMissing
import postprocessing
from dim_red_clustering import dim_red_clustering

# read dataset
ds_name = 'iris'
df = pd.DataFrame(arff.loadarff("datasets/" + ds_name + ".arff")[0])
print(
    "Loaded '" + ds_name + "' dataset! Use the CLI argument 'dataset' to choose from ['autos','iris','vote'] datasets!")
print("[STARTED] preprocessing\n")
if (ds_name == 'autos'):
    df.drop(columns=[df.columns[-1]], inplace=True)
    print("dropping old label 'symbolic-risk-factor'")
    print("choosing 'price' as new label")
    print("quantile binning 'price' into 2 bins (cheap/expensive)\n")
    binClass(df, 'quantile', 2)
decodeBytesAndReplaceMissing(df)

# ----------------------------------------Preprocessing stage-----------------------------------------------------------
# The way to treat Categorical features: onehot/labelled (for binary variables: onehot==labelled)
# The way to treat Missing Values: imputed/dropped
# Note: If the dataset has no categorical features, all the combinations output the same
#           ------------------------------------------------------------------------
#           |             imputed              |              dropped              |
# |---------|----------------------------------|-----------------------------------|
# | onehot  | type = 'onehot', impute = True   | type = 'onehot', impute = False   |
# |---------|----------------------------------|-----------------------------------|
# |labelled | type = 'labelled', impute = True | type = 'labelled', impute = False |
# |---------|----------------------------------|-----------------------------------|

features, true_labels, classes = preprocessing_pipeline(df=df, name=ds_name, type='onehot', impute=False)

# ----------------------------------Modify parameters of Dim Red Algorithms---------------------------------------------
pca_params = {
    'n_components': 2,
    'copy': True,
    'whiten': False,
    'svd_solver': 'auto',
    'tol': 0.0,
    'iterated_power': 'auto',
    'random_state': None
}

incr_pca_params = {
    'n_components': 2,
    'whiten': False,
    'copy': True,
    'batch_size': None
}

trunc_svd_params = {
    'n_components': 2,
    'algorithm': 'randomized',
    'n_iter': 5,
    'random_state': None,
    'tol': 0.0
}

pca_mai_params = {
    'n_components': 3,
    'print_details': False,
}

# ----------------------------------Modify parameters of Clustering Algorithms------------------------------------------

kmeans_params = {
    'n_clusters': 3,
    'distance': 'euclidean'
}

birch_params = {
    'threshold': 0.1,
    'branching_factor': 6,
    'n_clusters': 3,
    'compute_labels': True,
    'copy': True
}

# ----------------------------------Apply Algorithms with Dim Red Tech--------------------------------------------------
# dim_red = None, 'pca', 'incr_pca', 'trunc_svd', 'pca_mai'
# alg = birch, kmeans

dim_red = 'pca_mai'
alg = 'kmeans'

# ----------------------------------------------------------------------------------------------------------------------

if dim_red == 'pca':
    dim_red_params = pca_params
if dim_red == 'pca_mai':
    dim_red_params = pca_mai_params
if dim_red == 'trunc_svd':
    dim_red_params = trunc_svd_params
if dim_red == 'incr_pca':
    dim_red_params = incr_pca_params
if dim_red == None:
    dim_red_params = None

if alg == 'kmeans':
    alg_params = kmeans_params
if alg == 'birch':
    alg_params = birch_params

X_transformed, labels_pred = dim_red_clustering(features, true_labels, dim_red=dim_red,
                                                dim_red_params=dim_red_params, alg=alg, alg_params=alg_params)
print('Number of passed Dimension:',X_transformed.shape[1])
postprocessing.plot_results(X_transformed, labels_pred, true_labels, classes)
