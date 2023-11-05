import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import sklearn

def plot_data(data, labels):
# Method to plot data as PCA and ISOMAP representation with the first two components.
# input has to be: np.array for data without labels, and labels as label encoded classes (0,1,2,..)
    
    data = np.array(data)
    labels = np.array(labels)
    # Perform PCA to reduce data to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)

    # Perform Isomap for dimensionality reduction
    isomap = Isomap(n_components=2)
    X_isomap = isomap.fit_transform(data)

    # Create subplots for PCA and Isomap visualizations
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot PCA visualization
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=20)
    axes[0].set_title('PCA Visualization')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')

    # Plot Isomap visualization
    axes[1].scatter(X_isomap[:, 0], X_isomap[:, 1], c=labels, cmap='viridis', s=20)
    axes[1].set_title('Isomap Visualization')
    axes[1].set_xlabel('Isomap Component 1')
    axes[1].set_ylabel('Isomap Component 2')

    plt.show()

def plot_results(data, pred_labels, true_labels, classes_list):
# Method to plot data as PCA and ISOMAP representation, confusion matrix and table of scores.
# input has to be
    # data: np.array without labels
    # pred_labels: predicated labels as label encoded classes (0,1,2,..)
    # true_labels: true labels as label encoded classes (0,1,2,..)
    # classes_list: A list of the classes names which correspond to (0,1,2,..), for example: sklearn.preprocessing.LabelEncoder.classes_ (which was already trained before)
  pred_labels = np.array(pred_labels)
  labels_true = np.array(true_labels)
  labels_pred = fit_classes_name(pred_labels, true_labels)

  cf_matrix = sklearn.metrics.confusion_matrix(labels_true, labels_pred)

  ss = sklearn.metrics.silhouette_score(data, labels_pred)
  dbs = sklearn.metrics.davies_bouldin_score(data, labels_pred)
  ars = sklearn.metrics.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)
  table = [
    ["Silhouette Score", "{:.4f}".format(ss)],
    ["Davies Bouldin Score", "{:.4f}".format(dbs)],
    ["Adjusted Rand Score", "{:.4f}".format(ars)],
  ]

  # Perform PCA to reduce data to 2 dimensions
  pca = PCA(n_components=2)
  X_pca = pca.fit_transform(data)

  # Perform Isomap for dimensionality reduction
  isomap = Isomap(n_components=2)
  X_isomap = isomap.fit_transform(data)

  # Create subplots for PCA and Isomap visualizations
  fig, ax = plt.subplots(1, 4, figsize=(22, 5))

  # Plot PCA visualization
  ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=pred_labels, cmap='viridis', s=20)
  ax[0].set_title('PCA Visualization')
  ax[0].set_xlabel('Principal Component 1')
  ax[0].set_ylabel('Principal Component 2')

  # Plot Isomap visualization
  ax[1].scatter(X_isomap[:, 0], X_isomap[:, 1], c=pred_labels, cmap='viridis', s=20)
  ax[1].set_title('Isomap Visualization')
  ax[1].set_xlabel('Isomap Component 1')
  ax[1].set_ylabel('Isomap Component 2')

  cf = sns.heatmap(cf_matrix,xticklabels=classes_list, yticklabels=classes_list, annot=True, ax=ax[2], cbar=False)
  cf.set_ylabel('True Labels')
  cf.set_xlabel('Predicted Labels')
  cf.title.set_text('Confusion Matrix')


  tb= ax[3].table(cellText=table, loc='center', cellLoc='center')
  # Customize the appearance of the table
  tb.auto_set_font_size(False)
  tb.set_fontsize(12)
  tb.scale(1, 1.5) 
  ax[3].axis('off')

  plt.show()

def fit_classes_name(labels_pred, labels_true, return_mapping_array=False):
    """Method to reorder the grouping labels to fit the true labels in terms of naming, so a confusion matrix or other validation stuff can be used.
    Input values: labels_pred and labels_true arrays. They need to be the same length and have to have the same number of classes starting from 0."""

    labels_pred = np.array(labels_pred, dtype=np.int32)
    labels_true = np.array(labels_true, dtype=np.int32)

    # get number of different classes
    n_classes = len(np.unique(labels_true))

    if n_classes != len(np.unique(labels_pred)):
        print("ERROR! Number of classes do not match")
    if len(labels_pred) != len(labels_true):
        print("ERROR! Length of true and pred labels array does not match!")
    # create array with the index numbers
    index = np.arange(0, len(labels_true))
    # create helping arrays to store values
    n_same = np.zeros(n_classes)
    best = np.zeros(n_classes)

    # loop through every class in the pred labels
    for n in range(n_classes):
        index_ofclass_pred = index[(labels_pred == n)]
        # loop through every class in the true labels
        for i in range(n_classes):
            index_ofclass_true = index[(labels_true == i)]
            # store number of same indexes
            n_same[i] = len(np.intersect1d(index_ofclass_true, index_ofclass_pred))
            # print(index_ofclass_true)
        # store in array of n_class length the class i which fits the class n best
        best[n] = int(np.argmax(n_same))

    # create output array
    labels_new_pred = np.zeros(len(labels_true))

    # loop through number of classes
    for k in range(n_classes):
        # set labels of output array according to best array, by passing a index array
        labels_new_pred[index[(labels_pred == k)]] = best[k]
    

    if return_mapping_array:
        return labels_new_pred.astype(int), best.astype(int)

    return labels_new_pred.astype(int)
    # output: numpy array of reordered labels for labels_pred
  #loop through number of classes
  for k in range(n_classes):
    # set labels of output array according to best array, by passing a index array
    labels_new_pred[index[(labels_pred == k)]] = best[k]
