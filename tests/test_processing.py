import numpy as np
import pandas as pd
from stattools.data_processing import impute_vals, encode_categorical, log_transform

def test_impute_missing_values_mean():
    data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
    expected_result = np.array([[1, 2, 7.5], [4, 5, 6], [7, 8, 9]])
    result = impute_vals(data, strategy='mean')
    np.testing.assert_array_almost_equal(result, expected_result, decimal=1)

def test_impute_missing_values_median():
    data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]])
    expected_result = np.array([[1, 2, 9], [4, 8, 6], [7, 8, 9], [10, 11, 12]])
    result = impute_vals(data, strategy='median')
    np.testing.assert_array_almost_equal(result, expected_result, decimal=1)

def test_encode():
    df = pd.DataFrame({'Region': ['North', 'South', 'East', 'West']})
    result = encode_categorical(df, 'Region', 'onehot')
    assert 'Region_North' in result.columns
    assert 'Region_South' in result.columns
    assert 'Region_East' in result.columns
    assert 'Region_West' in result.columns

def test_encode_lab():
    df = pd.DataFrame({'Region': ['North', 'South', 'East', 'West']})
    result = encode_categorical(df, 'Region', 'label')
    assert set(result['Region']) == set(range(len(df['Region'].unique())))

def test_log_transform():
    df = pd.DataFrame({'Value': [1, 10, 100]})
    result = log_transform(df, 'Value')
    expected_result = np.log1p(df['Value'])
    np.testing.assert_array_almost_equal(result, expected_result, decimal=1)
