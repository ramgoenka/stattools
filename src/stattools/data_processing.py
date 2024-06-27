import numpy as np
import pandas as pd

def encode_categorical(df, column, method='onehot'):
    """
    Encode a categorical column in a DataFrame using the specified method.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the column to encode.
    column : str
        The name of the column to encode.
    method : str, optional
        The encoding method to use. Choose between 'onehot' (default) and 'label'.
        - 'onehot': Apply one-hot encoding using pd.get_dummies.
        - 'label': Apply label encoding by converting to category codes.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the encoded column.

    Raises
    ------
    ValueError
        If the specified method is not supported.
    """
    if method == 'onehot':
        return pd.get_dummies(df, columns=[column])
    elif method == 'label':
        df[column] = df[column].astype('category').cat.codes
        return df
    else:
        raise ValueError("Unsupported method. Choose 'onehot' or 'label'.")

def impute_vals(data, strategy='mean', fill_value=None):
    """
    Impute missing values in the given data using the specified strategy.

    Parameters
    ----------
    data : np.array
        The input data (2D array) containing missing values (NaN) to be imputed.
    strategy : str, optional
        The strategy to use for imputing values (default is 'mean').
        - 'mean': Replace NaN values with the mean of the column.
        - 'median': Replace NaN values with the median of the column.
        - 'constant': Replace NaN values with a specified fill_value.
    fill_value : scalar, optional
        The value to replace NaN with when the strategy is 'constant'.

    Returns
    -------
    np.array
        The data with imputed values.

    Raises
    ------
    ValueError
        If an unsupported strategy is specified or a column only contains NaN values.
    """
    data = np.array(data, dtype=np.float64)
    for i in range(data.shape[1]):
        if strategy == 'mean':
            value = np.nanmean(data[:, i])
        elif strategy == 'median':
            value = np.nanmedian(data[:, i])
        elif strategy == 'constant':
            value = fill_value
        else:
            raise ValueError("NOT SUPPORTED. TRY mean, median, constant")
        if np.isnan(value):
            raise ValueError("This column only has NaN")
        data[:, i] = np.where(np.isnan(data[:, i]), value, data[:, i])
    return data

def log_transform(df, column):
    """
    Apply a log transformation to a specified column in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the column to transform.
    column : str
        The name of the column to transform.

    Returns
    -------
    pd.Series
        A Series with the log-transformed values.
    """
    return np.log1p(df[column])

def binning_features(df, column, bins, labels, strategy='quantile'):
    """
    Bin values of a specified column into discrete intervals using the specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the column to bin.
    column : str
        The name of the column to bin.
    bins : int or list of scalars
        If an integer, defines the number of equal-width or quantile bins.
        If a list, defines the bin edges.
    labels : list of str
        Labels for the resulting bins.
    strategy : str, optional
        The binning strategy to use (default is 'quantile').
        The options are as follows - "quantile": Bin into quantile intervals using pd.qcut OR 
        "fixed": Bin into fixed-width intervals using pd.cut.

    Returns
    -------
    pd.DataFrame
        The DataFrame with a new binned column.

    Raises
    ------
    ValueError
        If an unsupported strategy is specified.
    """
    if strategy == 'quantile':
        df[column + '_binned'] = pd.qcut(df[column], q=bins, labels=labels)
    elif strategy == 'fixed':
        df[column + '_binned'] = pd.cut(df[column], bins=bins, labels=labels)
    else:
        raise ValueError("Unsupported strategy. Use 'quantile' or 'fixed'.")
    return df
