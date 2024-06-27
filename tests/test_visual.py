from stattools.visual import mpp, corr_mat
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

def test_mpp():
    performance_dict = {'Accuracy': 0.90, 'Precision': 0.85, 'Recall': 0.88, 'F1-Score': 0.86}
    try:
        mpp(performance_dict, title='Test Model Performance', kind='bar')
        plt.close()
        assert True 
    except Exception as e:
        assert False, f"MPP failed: {e}"
    try:
        mpp(performance_dict, title='MPP TEST', kind='line')
        plt.close()
        assert True
    except Exception as e:
        assert False, f"MPP failed: {e}"

def test_cor_mat():
    np.random.seed(0)
    df = pd.DataFrame(np.random.rand(10, 5), columns=list('ABCDE'))
    try:
        corr_mat(df, title='Test Correlation Matrix')
        plt.close() 
        assert True
    except Exception as e:
        assert False, f"Correlation matrix plotting failed: {e}"