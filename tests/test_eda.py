import pandas as pd
from stattools.eda import summary_statistics

def test_summary_statistics():
    """
    Test the summary_statistics function with a sample DataFrame.
    """
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1]
    })
    result = summary_statistics(data)
    expected_columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'mode']
    assert all(col in result.columns for col in expected_columns), "Not all statistical measures are present."
    assert result.loc['A', 'mean'] == 3, "Mean calculation for column A is incorrect."
    assert result.loc['B', 'mode'] == 1, "Mode calculation for column B is incorrect."
