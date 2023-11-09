import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, KBinsDiscretizer
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA

def moveClassToEnd(df: pd.DataFrame, className: str):
    '''moves the column with the specified name to the end of the dataframe'''
    labels = df.loc[:,className].to_numpy()
    df.drop(columns=[className],inplace=True)
    df.insert(len(df.columns),className,labels)

def binClass(df: pd.DataFrame, strategy='uniform', bins=2):
    '''bins the last column of the dataframe'''
    className = df.columns[-1]
    vals = np.transpose(np.array([df.loc[:,className].to_numpy(dtype=int)]))
    binned = np.transpose(KBinsDiscretizer(n_bins=bins,encode='ordinal',strategy=strategy).fit_transform(vals))[0]
    df.drop(columns=[className],inplace=True)
    df.insert(len(df.columns),className,binned)

def decodeBytesAndReplaceMissing(df: pd.DataFrame):
    '''decodes byte types to strings and replaces "?" with np.NaN'''
    for col, dtype in df.dtypes.items():
        if dtype == object:
            df[col] = df[col].apply(lambda x: x.decode("utf-8"))
    df.replace('?',np.NaN,inplace=True)

def normalizeNumeric(df: pd.DataFrame, scaler=MinMaxScaler()) -> pd.DataFrame:
    '''selects the numeric features and normalizes them'''
    if len(df.select_dtypes(include = ['float64']).columns) != 0:
        # store numerical features column names
        numerical_names = df.select_dtypes(include = ['float64']).columns
        # store numerical features normalized and assign numerical_names to columns headers
        features_numeric = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include = ['float64'])))
        features_numeric.columns = numerical_names
        return features_numeric
    return np.asarray([])

def getFirstNotNan(col):
    '''returns the first not null value in an array'''
    for val in col:
        if not pd.isnull(val):
            return val

def replaceNaNs(arr: np.ndarray, rep) -> np.ndarray:
    '''replaces NaNs in a 1d array'''
    ret = []
    for val in arr:
        if pd.isnull(val):
            ret.append(rep)
        else:
            ret.append(val)
    return np.array(ret)

def labelEncode(arr: np.ndarray) -> np.ndarray:
    '''label encodes a 1d array'''
    enc = LabelEncoder()
    enc.fit(arr)
    return enc.transform(arr), enc.classes_

def labelEncodeInt(arr: np.ndarray, ) -> np.ndarray:
    return LabelEncoder().fit_transform(replaceNaNs(arr, getFirstNotNan(arr)))

def encodeCategorical(df: pd.DataFrame, type='one-hot') -> pd.DataFrame:
    '''encodes the categorical features and preservers NaNs, either with one-hot or label encoding'''
    if len(df.select_dtypes(include = ['object']).columns) != 0:
        if type == 'label':
            # store categorical features column names
            categorical_names = df.select_dtypes(include = ['object']).columns
            encoded = []
            # encode each column, and place back NaNs after encoding
            for c in categorical_names:
                col = df.loc[:,c]
                enc = labelEncodeInt(col).tolist()
                for i in range(len(col)):
                    if pd.isnull(col[i]):
                        enc[i] = np.NaN
                encoded.append(enc)
            encoded = np.transpose(np.array(encoded))
            features_categ = pd.DataFrame(encoded)
            features_categ.columns = categorical_names
            return features_categ
        else:
            # create a variable to store the categorical features and use it in order to have de NaNs back
            features_to_encode = df.select_dtypes(include = ['object'])
            categories = []
            for col in features_to_encode.columns:
                s = features_to_encode[col]
                categories.append(s[s.notna()].unique())
            # encode the features by means of the parameter categories
            enc = OneHotEncoder(categories = categories, handle_unknown = 'ignore', drop = 'if_binary')
            features_categ = pd.DataFrame(enc.fit_transform(features_to_encode).toarray(), columns = enc.get_feature_names_out())
            # to have the NaNs back by comparing features_categ and features_to_encode variables
            for col in range(0,len(features_to_encode.columns)):
                for row in range(0,len(features_to_encode.iloc[:,col])):
                    if pd.isnull(features_to_encode.iloc[row,col]):
                        features_categ.iloc[row,col] = features_to_encode.iloc[row,col]
            return features_categ
    return np.array([])

def imputeMissing(df: pd.DataFrame) -> pd.DataFrame:
    '''imputes the missing values with KNN imputer, considering only the closest neighbour'''
    imputer = KNNImputer(n_neighbors=1)
    features_imputed = pd.DataFrame(imputer.fit_transform(df))
    features_imputed.columns = df.columns
    return features_imputed

def dropMissing(df: pd.DataFrame, threshold = 0.2, labels = None) -> pd.DataFrame:
    '''drops every column where more values are missing than the defined threshold, and drops all datapoints with missing values after'''
    features_dropped = df.copy()
    for col in df.columns:
        if df[col].isna().sum() > len(df)*threshold:
            features_dropped.drop(col, axis=1, inplace=True)
    if (labels is not None):
        features_dropped.insert(len(features_dropped.columns),"___labels___",labels)
    features_dropped.dropna(inplace=True)
    if (labels is not None):
        labels_dropped = features_dropped.iloc[:,-1].to_numpy()
        features_dropped.drop(columns=['___labels___'],inplace=True)
        return features_dropped, labels_dropped
    return features_dropped

def reduceDimensionsFor2dVisualization(dropped: pd.DataFrame, imputed: pd.DataFrame):
    """reduces the dimensionality of the feature space to 2 with pca (does the analysis on the dropped df)
    \nreturns a tuple of:
    \n• transformed dropped df
    \n• transformed imputed df
    \n• transformer for dropped datapoint(s)
    \n• transformer for imputed datapoint(s)"""
    pca = PCA(n_components=2).fit(dropped)

    #get dropped columns
    dropped_columns = list(set(imputed.columns).difference(set(dropped.columns)))
    #get dropped column indexes
    dropped_indexes = [imputed.columns.get_loc(c) for c in dropped_columns]
    dropped_indexes.sort()
    #do the pca with the fitted function on the dropped set
    pca_imputed = pca.transform(imputed.drop(dropped_columns, axis=1))
    pca_dropped = pca.transform(dropped)
    #drop features of datapoint(s)
    def dropStuff(a):
        a = np.array(a)
        if len(a.shape) == 1:
            a = np.array([a])
        shift = 0
        for idx in dropped_indexes:
            a = np.delete(a, idx-shift, 1)
            shift += 1
        if len(a) == 1:
            return a[0]
        return a
    transform_imputed = lambda a : pca.transform(dropStuff(a))
    transform_dropped = lambda a : pca.transform(a)
    return (pca_dropped, pca_imputed, transform_dropped, transform_imputed)


# -----------------------------------------Preprocessing Pipeline Wrapper------------------------------------------------
# The aim of this method is to abstract the preprocessing pipeline performed on the datasets. It encapsulates the use of
# all the previous functions

def preprocessing_pipeline(df: pd.DataFrame, name: str, type='onehot', impute=False, drop_ratio=0.2):
    """ The aim of this method is to abstract and encapsulate the preprocessing pipeline performed on the datasets.

    Input:
        - df (dataframe): with feature and classes columns
        - type (str): {onehot, labelled} It changes the output. Affects only categorical values. Numerical are always normalized
        - impute (bool): It changes the output. Output impute or dropped samples with missing values
        - drop_ratio (float [0,1]): It changes the output. To drop features that exceed drop_ratio (When type = 'imputed',
            does not affect the output

    Output: Feature dataset and labels column
    """

    # split labels and features
    true_labels, classes = labelEncode(df.iloc[:, -1].to_numpy())
    if name == 'autos':
        classes = ['cheap', 'expensive']
    unique_labels = list(set(true_labels))
    features_raw = df.iloc[:, :-1]

    features_numeric = normalizeNumeric(features_raw)
    features_onehot = encodeCategorical(features_raw, 'one-hot')
    features_labelled = encodeCategorical(features_raw, 'label')

    if 0 != features_onehot.shape[0]:
        t = len(features_onehot.columns) - len(features_labelled.columns)
        if t > 0: print("one-hot encoding inflated the feature space's dimensionality by " + str(t) + "!\n")

    # join normalized numercial and categorical one-hot encoded features
    if 0 != features_numeric.shape[0] and 0 != features_onehot.shape[0]:
        features_concat_onehot = features_onehot.join(features_numeric, how='right')
    elif 0 != features_numeric.shape[0]:
        features_concat_onehot = features_numeric
    else:
        features_concat_onehot = features_onehot

    # join normalized numercial and categorical label encoded features
    if 0 != features_numeric.shape[0] and 0 != features_labelled.shape[0]:
        features_concat_labelled = features_labelled.join(features_numeric, how='right')
    elif 0 != features_numeric.shape[0]:
        features_concat_labelled = features_numeric
    else:
        features_concat_labelled = features_labelled

    features_labelled_imputed = imputeMissing(features_concat_labelled)
    features_labelled_dropped, true_labels_dropped = dropMissing(features_concat_labelled,
                                                                 labels=true_labels, threshold=drop_ratio)
    features_onehot_imputed = imputeMissing(features_concat_onehot)
    features_onehot_dropped = dropMissing(features_concat_onehot, threshold=drop_ratio)

    print("the dimensionality of the one-hot encoded dataset is " + str(len(features_labelled_imputed.columns)) + "\n")
    print("the dimensionality of the label encoded dataset is " + str(len(features_labelled_imputed.columns)) + "\n")
    n = features_concat_labelled.isna().sum().sum()
    t2 = sum([True for _, row in features_concat_labelled.iterrows() if any(row.isnull())])
    if n > 0: print("imputed " + str(n) + " values across " + str(t2) + " datapoints!\n")
    n = len(features_concat_labelled.columns) - len(features_labelled_dropped.columns)
    if n > 0: print("dropped " + str(n) + " features because more than " + str(
        round(drop_ratio * 100)) + "% values of the feature were missing!\n")
    n = len(features_concat_labelled) - len(features_labelled_dropped)
    if n > 0: print("dropped " + str(n) + " datapoints because they had missing values!\n")

    if impute:
        if type == 'onehot':
            return features_onehot_imputed, true_labels, classes
        if type == 'labelled':
            return features_onehot_imputed, true_labels, classes
    else:
        if type == 'onehot':
            return features_onehot_dropped, true_labels_dropped, classes
        if type == 'labelled':
            return features_onehot_dropped, true_labels_dropped, classes


