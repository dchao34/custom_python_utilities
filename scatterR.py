import numpy as np
import matplotlib.pyplot as plt
from plot_format import create_single_figure, format_axes

# Default scatter plot color scheme
color = [ '#53ACCA', '#FFC466', '#FFAAAA', '#98D68E', '#CE89BE']

# Create a mask to indicate which array elements should be included in the
# subsampling.
# length: The length of the masked array to generate. This should agree with
#         the length of the array to index into.
# fraction: Fraction of masked elements set to True. This corresponds to the
#           fraction of the original array we would like to subsample.
def generate_subsample_mask(length, fraction=None):
    m = None

    # If no fraction is specified, assume it's 1.
    if not fraction:
        m = np.ones(length, dtype=bool)

    else:
        m = np.zeros(length, dtype=bool)
        pts = int(length * fraction)
        m[:pts] = True

        # Shuffles the array in place.
        np.random.shuffle(m)

    return m

# Create a scatter plot. Wraps pyplot.scatter for the actual plotting.
def scatterR(x1, x2, undersample=None,
             ax=None,
             s=80, linewidth=1.7,
             facecolor='none', edgecolor=None,
             xlabel=None, ylabel=None, title=None,
             axislabel_fontsize=20):
    """
    This is a wrapper for pyplot.scatter, but limits/extends its functionality
    to those that are relevant for cluster visualization.

    x1 and x2 are the two features that are plotted against each other. To
    plot a single cluster, simply provide 1D ndarrays for x1 and x2. To plot
    multiple clusters, provide lists of 1D ndarrays, where each element
    corresponds to one cluster.

    Parameters
    ----------
    x1 : (n,) array or list of (n,) arrays. This is the first feature axis.
         A single ndarray plots a single cluster while a list of them plots
         multiple. If a list is provided, x2 must also be a list of ndarrays
         that has the same number of elements as x1.

    x2 : (n,) array or list of (n,) arrays. This is the second feature axis.

    undersample: Float or a list of floats. When a single float is provided,
                 it is the fraction of data points that should be displayed.
                 The points that are displayed are sampled randomly. If a list
                 is provided, then each float is the fraction for each cluster
                 that should be displayed; in this case, the list length must
                 match x1 and x2. Default: None (no undersampling).

    ax : This is the axes for which to draw the scatterplot.
         pyplot.gca() is used if none is provided.

    s : This is the markersize. Same as the `s` parameter for pyplot.scatter.
        Default: 80

    linewidth : Linewidth for the marker circle. Default: 1.7.

    facecolor : Marker facecolor. Default: None (no fill).

    edgecolor : Marker line color. Default: None, which is the color rotation set
               by this module.

    xlabel : X axis label.

    ylabel : Y axis label.

    title : Plot title.

    axislabel_fontsize: Fontsize of axis tick label. Default: 20.

    Returns
    -------
    This function does not return anything.
    (In contrast to the original `pyplot.scatter()` function)

    """

    # Get Axes object and add labels.
    if not ax: ax = format_axes(plt.gca())
    if xlabel: ax.set_xlabel(xlabel, fontsize=axislabel_fontsize)
    if ylabel: ax.set_ylabel(ylabel, fontsize=axislabel_fontsize)
    if title: ax.set_title(title, fontsize=axislabel_fontsize)

    # Color coding of the scatter plot
    if not edgecolor: edgecolor=color

    # Preprocess the features and undersampling parameters into lists. This is
    # to accomodate for multi-category plotting.
    if not isinstance(x1, (list, tuple)): x1, x2 = [ x1 ], [ x2 ]
    if not isinstance(undersample, (list, tuple)):
        undersample = [ undersample ] * len(x1)

    # Iteratively plot each category
    for i in range(len(x1)):

        x1_i, x2_i = x1[i], x2[i]

        # Subsample the data for plotting purposes.
        m = generate_subsample_mask(x1_i.shape[0], undersample[i])
        x1_i, x2_i = x1_i[m], x2_i[m]

        # Use pyplot.scatter to plot.
        ax.scatter(x1_i, x2_i,
                   s=s, linewidth=linewidth,
                   facecolor=facecolor,
                   edgecolor=edgecolor[i % len(edgecolor)])

# Wrapper for scatterR for structured arrays.
def rec_scatterR(X, Y,
                 feature1_name, feature2_name,
                 categories=None,
                 **kwargs):
    """
    This is a convenience wrapper for using scatterR with structured arrays.

    Given a structured array X and 1D ndarray Y with category labels,
    make a scatter plot that contains one cluster for each specified category.

    Parameters
    ----------
    X : Structured ndarray. It contains any number of named columns.

    Y : 1D ndarray. Element j contains the category label for row j in X.

    feature1_name : The field entry name in X that should be the first
                    feature axis in the scatter plot.

    feature2_name : The field entry name in X that should be the second
                    feature axis in the scatter plot.

    categories : List of categories to be included in the scatter plot. Each
                 category gets a cluster. List entries must be the same as the
                 labels that occur in Y. By default, all possible labels in Y
                 gets plotted.

    **kwargs are keyword arguments for scatterR. The notable ones are listed
    below; for the others, please see scatterR.

    *undersample* : Float or a list of floats. Fraction of events to display in
                    the scatter plot. If a list is provided, its length must
                    be the same as `categories`.

    *s* : This is the markersize. Default: 80

    *linewidth* : Linewidth for the marker circle. Default: 1.7.

    *edgecolor* : Marker line color. Default: None, which is the color rotation set
                  by this module.

    *axislabel_fontsize* : Fontsize of axis tick label. Default: 20.

    Returns
    -------
    This function does not return anything.
    (In contrast to the original `pyplot.scatter()` function)

    """

    # Get the actual feature arrays.
    x1, x2 = X[feature1_name], X[feature2_name]

    # If none given, then plot all possibilities.
    if not categories:
        categories = np.unique(Y)

    # Assemble x1 and x2 to be passed into scatterR. One list element for each
    # listed category.
    x1_list, x2_list = [], []
    for i, l in enumerate(categories):
        subset = (Y == l)
        x1_l, x2_l = x1[subset], x2[subset]
        x1_list.append(x1_l)
        x2_list.append(x2_l)

    scatterR(x1_list, x2_list, **kwargs)


if __name__ == "__main__":

    (x1, y1) = np.random.normal(3, 1, 500), np.random.normal(3, 2, 500)
    (x2, y2) = np.random.normal(-3, 2, 500), np.random.normal(-3, 1, 500)
    (x3, y3) = np.random.normal(-3, 2, 500), np.random.normal(3, 1, 500)
    (x4, y4) = np.random.normal(6, 2, 500), np.random.normal(3, 1, 500)
    (x5, y5) = np.random.normal(6, 2, 500), np.random.normal(-3, 1, 500)

    fig, ax = create_single_figure(figsize=(10, 10))
    scatterR([x1, x2, x3, x4, x5],
             [y1, y2, y3, y4, y5],
             undersample=[0.01, 0.1, None, 0.2, None],
             ax=ax, xlabel='feature 1', ylabel='feature 2')
    plt.show()
