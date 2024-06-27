import pandas as pd
def summary_statistics(data):
    """
    Presents a summary statistics report for a given dataset.

    Parameters
    ----------
    data : DataFrame
        The input dataset for which summary statistics are to be computed.

    Returns
    -------
    summary_stats : DataFrame
        A DataFrame containing the summary statistics for each variable in the input dataset.
        The summary statistics include: Mean, Median, Standard deviation, First quartile, Third quartile, Maximum value, Mode.
    """
    summary_stats = data.describe().T
    summary_stats['mode'] = data.mode().iloc[0]
    return summary_stats