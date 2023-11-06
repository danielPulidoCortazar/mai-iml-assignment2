import pandas as pd
from scipy.io import arff
import preprocessing
import postprocessing
from dim_red_clustering import dim_red_clustering

# ----------------------------------------Preprocessing stage-----------------------------------------------------------
# read dataset
dataset = 'iris'
df = pd.DataFrame(arff.loadarff("datasets/" + dataset + ".arff")[0])
print(
    "Loaded '" + dataset + "' dataset! Use the CLI argument 'dataset' to choose from ['autos','iris','vote'] datasets!")

print("[STARTED] preprocessing\n")
if (dataset == 'autos'):
    df.drop(columns=[df.columns[-1]], inplace=True)
    print("dropping old label 'symbolic-risk-factor'")
    print("choosing 'price' as new label")
    print("quantile binning 'price' into 2 bins (cheap/expensive)\n")
    preprocessing.binClass(df, 'quantile', 2)
preprocessing.decodeBytesAndReplaceMissing(df)

# split labels and features
true_labels, classes = preprocessing.labelEncode(df.iloc[:, -1].to_numpy())
if dataset == 'autos':
    classes = ['cheap', 'expensive']
unique_labels = list(set(true_labels))
features_raw = df.iloc[:, :-1]

features_numeric = preprocessing.normalizeNumeric(features_raw)
features_onehot = preprocessing.encodeCategorical(features_raw, 'one-hot')
features_label = preprocessing.encodeCategorical(features_raw, 'label')

if type(features_onehot) != type([]):
    t = len(features_onehot.columns) - len(features_label.columns)
    if t > 0: print("one-hot encoding inflated the feature space's dimensionality by " + str(t) + "!\n")

# join normalized numercial and categorical one-hot encoded features
if type(features_numeric) != type([]) and type(features_onehot) != type([]):
    features_concat_onehot = features_onehot.join(features_numeric, how='right')
elif type(features_numeric) != type([]):
    features_concat_onehot = features_numeric
else:
    features_concat_onehot = features_onehot

# join normalized numercial and categorical label encoded features
if type(features_numeric) != type([]) and type(features_label) != type([]):
    features_concat_label = features_label.join(features_numeric, how='right')
elif type(features_numeric) != type([]):
    features_concat_label = features_numeric
else:
    features_concat_label = features_label

thr = 0.2
onehot_imputed = preprocessing.imputeMissing(features_concat_onehot)
onehot_dropped, dropped_labels = preprocessing.dropMissing(features_concat_onehot, labels=true_labels, threshold=thr)
label_imputed = preprocessing.imputeMissing(features_concat_label)
label_dropped = preprocessing.dropMissing(features_concat_label, threshold=thr)

print("the dimensionality of the one-hot encoded dataset is " + str(len(onehot_imputed.columns)) + "\n")
print("the dimensionality of the label encoded dataset is " + str(len(label_imputed.columns)) + "\n")
t = features_concat_label.isna().sum().sum()
t2 = sum([True for _, row in features_concat_label.iterrows() if any(row.isnull())])
if t > 0: print("imputed " + str(t) + " values across " + str(t2) + " datapoints!\n")
t = len(features_concat_label.columns) - len(label_dropped.columns)
if t > 0: print("dropped " + str(t) + " features because more than " + str(
    round(thr * 100)) + "% values of the feature were missing!\n")
t = len(features_concat_label) - len(label_dropped)
if t > 0: print("dropped " + str(t) + " datapoints because they had missing values!\n")

# ----------------------------------Modify parameters of Dim Red Algorithms---------------------------------------------
pca_params = {
    'n_components': None,
    'copy': True,
    'whiten': False,
    'svd_solver': 'auto',
    'tol': 0.0,
    'iterated_power': 'auto',
    'n_oversamples': 10,
    'power_iteration_normalizer': 'auto',
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

# ----------------------------------Modify parameters of Clustering Algorithms---------------------------------------------

kmeans_params = {
    'n_clusters': 3,
    'distance': 'manhattan'
}

birch_params = {
    'threshold': 0.3,
    'branching_factor': 50,
    'n_clusters': 3,
    'compute_labels': True,
    'copy': True
}


#----------------------------------Apply Algorithms with Dim Red Tech---------------------------------------------------

# Be aware that you have to change n_clusters in clustering algorithms params when changing datasets!!
X_transformed, labels_pred = dim_red_clustering(onehot_imputed, true_labels, dim_red='pca',
                                                dim_red_params=pca_params, alg='birch', alg_params=birch_params)


postprocessing.plot_results(X_transformed, labels_pred, true_labels, classes)


