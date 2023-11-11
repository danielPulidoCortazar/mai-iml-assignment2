import pandas as pd
from scipy.io import arff
from preprocessing import preprocessing_pipeline, binClass, decodeBytesAndReplaceMissing
import postprocessing
from dim_red_clustering import dim_red_clustering

# read dataset
ds_name = 'vote'
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

features, true_labels, classes = preprocessing_pipeline(df=df, name=ds_name,
                                                        type='labelled', impute=True)

# ----------------------------------Modify parameters of Dim Red Algorithms---------------------------------------------
pca_params = {
    'n_components': None,
    'copy': True,
    'whiten': False,
    'svd_solver': 'auto',
    'tol': 0.0,
    'iterated_power': 'auto',
    'random_state': None
}

incr_pca_params = {
    'n_components': None,
    'whiten': False,
    'copy': True,
    'batch_size': None
}

trunc_svd_params = {
    'n_components': 4,
    'algorithm': 'randomized',
    'n_iter': 5,
    'n_oversamples': 10,
    'power_iteration_normalizer': 'auto',
    'random_state': None,
    'tol': 0.0
}

pca_mai_params = {
    'print_details': True,
}

# ----------------------------------Modify parameters of Clustering Algorithms---------------------------------------------

kmeans_params = {
    'n_clusters': 2,
    'distance': 'euclidean'
}

birch_params = {
    'threshold': 0.5,
    'branching_factor': 8,
    'n_clusters': 2,
    'compute_labels': True,
    'copy': True
}

# ----------------------------------Apply Algorithms with Dim Red Tech--------------------------------------------------
# dim_red = pca, incr_pca, trunc_svd, pca_mai
# alg = birch, kmeans

X_transformed, labels_pred = dim_red_clustering(features, true_labels, dim_red=None,
                                                dim_red_params=None, alg='kmeans', alg_params=kmeans_params)

postprocessing.plot_results(X_transformed, labels_pred, true_labels, classes)