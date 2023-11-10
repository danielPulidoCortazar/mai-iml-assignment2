# basic imports
import math
import argparse
import warnings
warnings.filterwarnings('ignore')

# import allowed tools
from scipy.io import arff
import pandas as pd
from sklearn.manifold import Isomap

# import visualization tools
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', top=False, bottom=False, labeltop=False, labelbottom=False)
matplotlib.rc('ytick', left=False, right=False, labelleft=False, labelright=False)
matplotlib.rc('axes', titlesize=9)

# import from scratch sripts
import preprocessing
from dim_red_clustering import dim_red_clustering
from pca_mai import my_pca

# cli args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d","--dataset",default="autos",help="choose a dataset from ['autos','iris','vote']")
parser.add_argument("-t","--preproc_type",default="labelled",help="choose a preprocessing type from ['labelled','onehot']")
parser.add_argument("-i","--preproc_impute",default="false",help="choose a preprocessing imputation from ['false','true']")
args = parser.parse_args()
config = vars(args)

# read dataset
dataset = config['dataset']
df = pd.DataFrame(arff.loadarff("datasets/"+dataset+".arff")[0])
print("Loaded '"+dataset+"' dataset! Use the CLI argument 'dataset' to choose from ['autos','iris','vote'] datasets!")

print("[STARTED] preprocessing\n")
if (dataset == 'autos'):
    df.drop(columns=[df.columns[-1]],inplace=True)
    print("dropping old label 'symbolic-risk-factor'")
    print("choosing 'price' as new label")
    print("quantile binning 'price' into 2 bins (cheap/expensive)\n")
    preprocessing.binClass(df,'quantile',2)

features, true_labels, classes = preprocessing.preprocessing_pipeline(df=df,name=dataset,type=config['preproc_type'],impute=config['preproc_impute']=='true')

gridX = 7
gridY = 4
fig, axes = plt.subplots(gridY, gridX)
page = 1
fig.suptitle(dataset+' dataset (page '+str(page)+')')
fig.tight_layout()
cnt = 0
def nextAxes():
    global cnt, page, fig, axes
    if cnt >= gridX*gridY:
        fig, axes = plt.subplots(gridY, gridX)
        page += 1
        fig.suptitle(dataset+' dataset (page '+str(page)+')')
        fig.tight_layout()
        cnt = 0
    y = math.floor(cnt/gridX)
    x = cnt - y*gridX
    cnt += 1
    ret = axes[y,x]
    return ret

print("[FINISHED] preprocessing\n")

dim_red_arr = {
    'no_red': {},
    'pca': {
        'n_components': None,
        'copy': True,
        'whiten': False,
        'svd_solver': 'auto',
        'tol': 0.0,
        'iterated_power': 'auto',
        'random_state': None
    },
    'incr_pca': {
        'n_components': None,
        'whiten': False,
        'copy': True,
        'batch_size': None
    },
    'trunc_svd': {
        'n_components': 3,
        'algorithm': 'randomized',
        'n_iter': 5,
        'random_state': None,
        'tol': 0.0
    },
    'pca_mai': {
        'print_details': False,
    },
}
name = {
    'pca': 'sklearn pca',
    'incr_pca': 'sklean incremental pca',
    'trunc_svd': 'truncated svd',
    'pca_mai': 'pca from scratch (ours)',
    'no_red': 'no reduction'
}

print("Printing PCA intermediate steps!")

X_transformed = my_pca(features, True)
isomap = Isomap(n_components=2)
X_isomap = isomap.fit_transform(features)

print("Starting experiments now, when all are finished, all diagrams will be displayed!")

for key in dim_red_arr:
    if key == 'no_red': continue
    dim_red_params = dim_red_arr[key]
    X_transformed, _ = dim_red_clustering(features,true_labels,key,dim_red_params,'')
    a = nextAxes()
    p = sns.scatterplot(x=X_transformed[:,0],y=X_transformed[:,1],hue=true_labels,legend=None,ax=a)
    p.set(title='true labels '+name[key])

a = nextAxes()
p = sns.scatterplot(x=X_isomap[:,0],y=X_isomap[:,1],hue=true_labels,legend=None,ax=a)
p.set(title='true labels isomap')

birch_params = {
    'autos': {
        'threshold': 0.1,
        'branching_factor': 50,
        'n_clusters': 2,
    },
    'iris': {
        'threshold': 0.2,
        'branching_factor': 50,
        'n_clusters': 3,
    },
    'vote': {
        'threshold': 0.5,
        'branching_factor': 8,
        'n_clusters': 2,
    },
}[dataset]

for key in dim_red_arr:
    dim_red_params = dim_red_arr[key]
    _, labels_pred = dim_red_clustering(features,true_labels,key,dim_red_params,'birch',birch_params)
    a = nextAxes()
    p = sns.scatterplot(x=X_transformed[:,0],y=X_transformed[:,1],hue=labels_pred,legend=None,ax=a)
    p.set(title='(PCA) birch on '+name[key])
    a = nextAxes()
    p = sns.scatterplot(x=X_isomap[:,0],y=X_isomap[:,1],hue=labels_pred,legend=None,ax=a)
    p.set(title='(ISOMAP) birch on '+name[key])

kmeans_params = {
    'autos': {
        'n_clusters': 2,
        'distance': 'euclidean'
    },
    'iris': {
        'n_clusters': 3,
        'distance': 'euclidean'
    },
    'vote': {
        'n_clusters': 2,
        'distance': 'euclidean'
    },
}[dataset]

for key in dim_red_arr:
    dim_red_params = dim_red_arr[key]
    _, labels_pred = dim_red_clustering(features,true_labels,key,dim_red_params,'kmeans',kmeans_params)
    a = nextAxes()
    p = sns.scatterplot(x=X_transformed[:,0],y=X_transformed[:,1],hue=labels_pred,legend=None,ax=a)
    p.set(title='(PCA) kmeans on '+name[key])
    a = nextAxes()
    p = sns.scatterplot(x=X_isomap[:,0],y=X_isomap[:,1],hue=labels_pred,legend=None,ax=a)
    p.set(title='(ISOMAP) kmeans on '+name[key])

plt.show()