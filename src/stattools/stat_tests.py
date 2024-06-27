import numpy as np
from scipy import stats

def t_test(data1, data2):
    """
    Perform a two-sample t-test to compare the means of two datasets.

    Parameters
    ----------
    data1 : array_like
        The first dataset as an array or list of numerical values.
    data2 : array_like
        The second dataset as an array or list of numerical values.

    Returns
    -------
    t_statistic : float
        The computed t-statistic, which measures the difference in means of
        the two datasets in terms of standard error.
    p_value : float
        The two-tailed p-value indicating the probability of observing the data
        if the null hypothesis is true (i.e., the means of the two datasets are the same).
    """
    mean1, mean2 = np.mean(data1), np.mean(data2)
    n1, n2 = len(data1), len(data2)
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    t_val = (mean1 - mean2) / np.sqrt(pooled_var * (1 / n1 + 1 / n2))
    df = n1 + n2 - 2
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_val), df))
    return t_val, p_value


def chi_sq(observed, expected):
    """
    Compute the Chi-Square statistic and p-value for goodness-of-fit.

    Parameters
    ----------
    observed : array-like
        The observed frequencies of each category.
    expected : array-like
        The expected frequencies of each category.

    Returns
    -------
    chi_square_stat : float
        The computed Chi-Square statistic.
    p_value : float
        The p-value calculated using the Chi-Square distribution.
    """
    chi_square_stat = np.sum((observed - expected)**2 / expected)
    degrees_of_freedom = len(observed) - 1
    simulated_chi_square = np.random.chisquare(degrees_of_freedom, 10000)
    p_value = np.sum(simulated_chi_square >= chi_square_stat) / 10000

    return chi_square_stat, p_value


def anova(*args):
    """
    Perform one-way ANOVA to test difference between means of groups.

    Parameters
    ----------
    *args : array-like
        Variable number of input arrays representing different groups.

    Returns
    -------
    F : float
        The computed F-statistic.
    p_value : float
        The p-value for the hypothesis test, indicating whether the observed
        differences between group means are statistically significant.
    """
    k = len(args)
    n = sum(len(group) for group in args)
    grand_mean = np.mean([item for group in args for item in group])
    ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in args)
    ss_within = sum(sum((item - np.mean(group)) ** 2 for item in group) for group in args)
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    F = ms_between / ms_within
    p_value = 1 - stats.f.cdf(F, dfn=k-1, dfd=n-k)
    return F, p_value

def pearson_corr(x, y):
    """
    Compute the Pearson correlation coefficient between two arrays.

    Parameters
    ----------
    x : array-like
        First input data array.
    y : array-like
        Second input data array.

    Returns
    -------
    correlation_coeff : float
        Pearson's correlation coefficient, ranging from -1 to +1.
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    covariance = np.mean((x - mean_x) * (y - mean_y))
    correlation_coeff = covariance / (std_x * std_y)
    return correlation_coeff