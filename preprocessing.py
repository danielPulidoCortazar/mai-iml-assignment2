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
    return []

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
    return []

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