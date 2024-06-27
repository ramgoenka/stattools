import seaborn as sns
import matplotlib.pyplot as plt
def corr_mat(data, title='Correlation Matrix', cmap='coolwarm', figsize=(10, 8)):
    """
    Create a correlation heatmap to identify relationships between features.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataset containing the features to be visualized in the correlation matrix.
    title : str, optional
        Title of the heatmap. Defaults to 'Correlation Matrix'.
    cmap : str, optional
        Colormap to use for visualization. Defaults to 'coolwarm'.
    figsize : tuple of (float, float), optional
        Size of the figure. Defaults to (10, 8).

    Returns
    -------
    None
        The function visualizes a heatmap using seaborn and matplotlib.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap=cmap)
    plt.title(title)
    plt.show()

def mpp(performance_dict, title='Model Performance', kind='bar'):
    """
    Visualize model performance metrics as bar or line charts.

    Parameters
    ----------
    performance_dict : dict
        Dictionary where keys are metric names and values are metric scores.
    title : str, optional
        Title of the plot. Defaults to 'Model Performance'.
    kind : {'bar', 'line'}, optional
        Type of plot to use. 'bar' creates a bar chart, 'line' creates a line plot. Defaults to 'bar'.

    Returns
    -------
    None
        The function visualizes a bar or line chart using matplotlib.
    
    Raises
    ------
    ValueError
        If the 'kind' parameter is not 'bar' or 'line'.
    """
    fig, ax = plt.subplots()
    metrics = list(performance_dict.keys())
    values = list(performance_dict.values())
    if kind == 'bar':
        ax.bar(metrics, values, color='skyblue')
    elif kind == 'line':
        ax.plot(metrics, values, marker='o', linestyle='-', color='skyblue')
    else:
        raise ValueError("Unsupported kind. Use 'bar' or 'line'.")
    ax.set_title(title)
    ax.set_ylabel('Value')
    ax.set_xlabel('Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
