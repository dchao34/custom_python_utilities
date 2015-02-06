import numpy as np
import matplotlib.pyplot as plt
from plot_format import create_single_figure, format_axes, hastie_colors

# Default scatter plot color scheme
color_single = [ 'gray' ]
color_wheel = hastie_colors

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
             marker_size=80, linewidth=1.7,
             edgecolor=None, facecolor='none',
             xlabel=None, ylabel=None, title=None,
             axislabel_fontsize=20,
             legend=False, legend_names=None,
             legend_loc=None, legend_ncol=None, legend_fontsize=20):
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

    marker_size : This is the markersize. Same as the `s` parameter for pyplot.scatter.
                  Default: 80

    linewidth : Linewidth for the marker circle. Default: 1.7.

    edgecolor : Marker line color. Default: None, which is the color rotation set
                by this module; gray for a single cluster, default color wheel
                otherwise.

    facecolor : Marker facecolor. Default: None (no fill; recommended).

    xlabel : X axis label.

    ylabel : Y axis label.

    title : Plot title.

    axislabel_fontsize: Fontsize of axis tick label. Default: 20.

    legend : Whether to include a legend. Default: False.

    legend_names : List of strings. Element i corresonds to the legend text
                   for item i in x1 and x2. Default: None, which gets converted to
                   numbers, one for each category.

    legend_loc : Legend location; same as *loc* keyword argument for pyplot.legend().
                 Default: None, which is 'best'.

    legend_ncol : Number of columns in the legend; same as *ncol* keyword argument for
                  pyplot.legend(). Default: One column for each category.

    legend_fontsize : Legend fontsize. Default: 20.

    Returns
    -------
    This function does not return anything.
    (In contrast to the original `pyplot.scatter()` function)

    """

    # Get Axes object and add labels.
    if not ax:
        ax = format_axes(plt.gca())
        ax.autoscale(tight=True)
    if xlabel: ax.set_xlabel(xlabel, fontsize=axislabel_fontsize)
    if ylabel: ax.set_ylabel(ylabel, fontsize=axislabel_fontsize)
    if title: ax.set_title(title, fontsize=axislabel_fontsize)

    # Preprocess the features and undersampling parameters into lists. This is
    # to accomodate for multi-category plotting.
    if not isinstance(x1, (list, tuple)): x1, x2 = [ x1 ], [ x2 ]
    if not isinstance(undersample, (list, tuple)):
        undersample = [ undersample ] * len(x1)

    # Color coding of the scatter plot
    if not edgecolor:
        if len(x1) == 1:
            edgecolor=color_single
        else:
            edgecolor=color_wheel
    else:
        if not isinstance(edgecolor, (list, tuple)): edgecolor = [ edgecolor ]

    # Iteratively plot each category
    legend_handles = []
    for i in range(len(x1)):

        x1_i, x2_i = x1[i], x2[i]

        # Subsample the data for plotting purposes.
        m = generate_subsample_mask(x1_i.shape[0], undersample[i])
        x1_i, x2_i = x1_i[m], x2_i[m]

        # Use pyplot.scatter to plot.
        handle = ax.scatter(x1_i, x2_i,
                            s=marker_size, linewidth=linewidth,
                            facecolor=facecolor,
                            edgecolor=edgecolor[i % len(edgecolor)])
        legend_handles.append(handle)

    # Build legend
    if legend:
        if legend_names is None: legend_names = map(str, range(len(x1)))
        if legend_loc is None: legend_loc = 'best'
        if legend_ncol is None: legend_ncol = len(x1)
        ax.legend(legend_handles, legend_names,
                  loc=legend_loc, ncol=legend_ncol,
                  scatterpoints=1, fontsize=legend_fontsize)

# Partition x1, x2 by Y.
#
# Input:
#
#     x1, x2 : 1D ndarrays of the same length.
#
#     Y : 1D ndarray of the same length as x1 and x2. Element i of Y is a label
#        for element i for both x1 and x2.
#
#     categories : List of labels from Y to include in the result.
#
# Output:
#
#     x1_list, x2_list : List of 1D ndarrays that partitions x1 and x2 by the
#                        labels in Y. Only those labels in `categories` is
#                        included in the list.
#
def partition_by_labels(x1, x2, Y=None, categories=None):

    x1_list, x2_list = [], []

    # If no Y is given, return a single partition containing all elements.
    if Y is None:
        x1_list, x2_list = [ x1 ], [ x2 ]
    else:
        # If no specific categories are given, then partition by all unique
        # labels in Y.
        if not categories:
            categories = np.unique(Y)

        # Perform the partition.
        for i, l in enumerate(categories):
            subset = (Y == l)
            x1_l, x2_l = x1[subset], x2[subset]
            x1_list.append(x1_l)
            x2_list.append(x2_l)

    return x1_list, x2_list

# Wrapper for scatterR for structured arrays.
def scatterRec(X, feature1_name, feature2_name,
               Y=None, categories=None,
               **kwargs):
    """
    This is a convenience wrapper for using scatterR with structured arrays.

    Given a structured array X and 1D ndarray Y with category labels,
    make a scatter plot that contains one cluster for each specified category.

    Parameters
    ----------
    X : Structured ndarray. It contains any number of named columns.

    feature1_name : The field entry name in X that should be the first
                    feature axis in the scatter plot.

    feature2_name : The field entry name in X that should be the second
                    feature axis in the scatter plot.

    Y : String or 1D ndarray. If string, the category of each record is assumed
        to be under the given field name. If an ndarray, it should have the same
        length as the structured array where element i is the category label for
        the ith record. Default: None. Treat everything as a single category.

    categories : If Y is given, then this is the list of categories to be included
                 in the scatter plot. The entries in this list must be an existing
                 entry in Y. Default: None; this gives every category in Y a cluster.

    **kwargs are keyword arguments for scatterR. The notable ones are listed
    below; for the others, please see scatterR.

    *undersample* : Float or a list of floats. Fraction of events to display in
                    the scatter plot. If a list is provided, its length must
                    be the same as `categories`.

    *marker_size* : This is the markersize. Default: 80

    *linewidth* : Linewidth for the marker circle. Default: 1.7.

    *edgecolor* : Marker line color. Default: None, which is the color rotation set
                  by this module.

    *axislabel_fontsize* : Fontsize of axis tick label. Default: 20.

    *legend* : Whether to include a legend. Default: False.

    *legend_names* : List of strings. Element i corresonds to the legend text
                   for item i in x1 and x2. In this case, should have the same number of
                   elements as `categories.
                   Default: None, which gets converted to numbers, one for each category.

    *legend_loc* : Legend location; same as *loc* keyword argument for pyplot.legend().
                 Default: None, which is 'best'.

    *legend_ncol* : Number of columns in the legend; same as *ncol* keyword argument for
                  pyplot.legend(). Default: None, which converts to the number of categories.

    *legend_fontsize* : Legend fontsize. Default: 20.

    Returns
    -------
    This function does not return anything.
    (In contrast to the original `pyplot.scatter()` function)

    """

    # Get the actual feature arrays.
    x1, x2 = X[feature1_name], X[feature2_name]

    # Partition by labels
    if isinstance(Y, str): Y = X[Y]
    x1_list, x2_list = partition_by_labels(x1, x2, Y, categories)

    # Plot by partitions
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
             undersample=[0.1, 0.1, None, 0.2, None],
             ax=ax, xlabel='feature 1', ylabel='feature 2',
             legend=True, legend_names=['zero', 'one', 'two', 'three', 'four'],
             legend_ncol=5, legend_loc='lower right', legend_fontsize=12)
    plt.show()
