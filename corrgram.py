import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from plot_format import format_axes
from scatterR import scatterR, partition_by_labels

# Create the plotting area. Return the figure object and a list of list of Axes
# objects. We can access the (i,j)th Axes object in axs[i][j].
def create_plotting_area(M, figsize=None, tight=None, tick_nbins=None):
    if not figsize:
        figsize=(10,10)

    fig = plt.figure(figsize=figsize)

    # Initiate the plotting grid. Don't leave any space between subplots.
    gs = gridspec.GridSpec(M, M, wspace=0.0, hspace=0.0)

    # Initiate all the Axes instances.
    axs = []
    for i in range(M):
        row = []
        for j in range(M):
            ax = plt.subplot(gs[i,j])

            # Adjust the number of axis ticks, but don't tamper with the
            # diagonal entries.
            if i != j:
                ax.locator_params(tight=tight, nbins=tick_nbins)

            row.append(ax)
        axs.append(row)
    return fig, axs

# Make scatter plots in the upper triangular entries.
def plot_scatter(axs, x, y=None, categories=None, **kwargs):

    it = itertools.combinations(range(len(x)), 2)
    while True:
        try:
            i, j = it.next()

            # Get the ndarrays that we would like to plot and partition by y
            x1, x2 = partition_by_labels(x[i], x[j], y, categories)

            # Plot scatter plot using scatterR. Note that x2 comes before x1.
            scatterR(x2, x1, ax=axs[i][j], **kwargs)
        except StopIteration:
            break

# Annotate the diagonal plots. Simply place the feature name in the center.
def plot_text(axs, names, diag_fontsize):
    if not names:
        return
    for i, name in enumerate(names):
        axs[i][i].text(0.5, 0.5, name, fontsize=diag_fontsize, ha='center', va='center')

# Make pie plots in the lower triangular entries.
def plot_pie(axs, x):

    it = itertools.combinations(range(len(x)), 2)
    while True:
        try:
            i, j = it.next()

            # Compute pearson's correlation coefficient.
            corr_coef = stats.pearsonr(x[i], x[j])[0]

            # Depending on the sign of the coefficient, we want to plot the pie with
            # different color/fill direction.
            sizes, colors = [], []
            if corr_coef > 0:
                sizes = [ 1 - corr_coef, corr_coef ]
                colors = [ '#FFFFFF', '#D46A6A' ]
            else:
                sizes = [ - corr_coef, 1 + corr_coef ]
                colors = [ '#4E648E', '#FFFFFF' ]

            # Plot the pie charts and tidy up the lines. Note the order of i,j
            # in axs element access.
            pie_wedges, pie_text = axs[j][i].pie(sizes, startangle=90, colors=colors)
            for wedge in pie_wedges: wedge.set_ec('gray')

        except StopIteration:
            break


# This function makes a final pass to make the corrgram plot readable.
# It is the last plotting function that is called before returning the figure
# back to the user.
def format_plot(axs, tick_fontsize=None):

    M = len(axs)
    it = itertools.product(range(M), range(M))
    while True:
        try:
            i, j = it.next()
            ax = axs[i][j]

            # Font size for the tick labels.
            if tick_fontsize:
                ax.tick_params(labelsize=tick_fontsize)

            # Remove axis and tick labels that get in the way.
            if i == 0 and j != i:
                ax.xaxis.tick_top()
                t = ax.get_xaxis().get_major_ticks()
                t[0].label2.set_visible(False)
                t[-1].label2.set_visible(False)
            else:
                ax.xaxis.set_visible(False)
            if j == M - 1 and j != i:
                ax.yaxis.tick_right()
                t = ax.get_yaxis().get_major_ticks()
                t[0].label2.set_visible(False)
                t[-1].label2.set_visible(False)
            else:
                ax.yaxis.set_visible(False)
        except StopIteration:
            break

# Main corrgram plotting function.
def corrgram(x, names,
             y=None, categories=None,
             figsize=None,
             tight=True,
             diag_fontsize=32,
             tick_fontsize=14, tick_nbins=5,
             **kwargs):
    """
    Plot a correlogram of ndarrys in list `x`. This was inspired by a
    similar function in R.

    `x` is a list of 1D ndarrays of equal length. `names` is a list of
    strings that provide a more descriptive names of each array.

    This function takes the `x` and `names` to make a grid of plots.
    The upper triagular elements are scatterplots between each pair of fields.
    The diagonal specifies the field names that are being plotted.
    The lower triagular elements are pie charts that indicates the
    pearson correlation coefficient.

    `y` is a 1D-array with the same length as each element in `x`. Element i
    in `y` is a label for the ith element in all ndarrays in `x`. When `y` is
    provided, the scatter plot will be color coded by how `y` partitions the
    elements in `x`. An additional `categories` parameter decides which labels
    are actually plotted in the scatter plot.


    Parameters
    ----------
    x : List of 1 dimension ndarray's of equal length.

    names : List of strings. These are aliases of the field names that will
            be printed in the diagonal elements. If None, the strings in
            `fields` will be printed. Default: None.

    y : 1D ndarray that indicates the label for elements in each member of `x`.
        Default: None.

    categories : If `y` is given, then this is the list of categories that will
                 actually be displayed in the scatter plot.
                 Default: None. In this case, display all unique labels in `y`.

    figsize : tuple (default=(10,10)).
              Size of the figure to create.

    tight : bool (default=True)
            Whether to make the scatter plot `tight`.
            See `axes.locator_params`'s `tight`.

    diag_fontsize : int (default=32)
                    Fontsize of the diagonal text labels.

    tick_nbins : int (default=5)
            Adjusts number of ticks in the scatter plot.
            See `axes.locator_params`'s `nbins`.

    tick_fontsize : int (default=14)
                    Fontsize of the axis tick labels.

    **kwargs : Keyword arguments for scatterR. Notable ones include:

        *undersample*: float (default=None).
                       The proportion of points to plot in the scatter plot.

        *marker_size*: float (default=None)
                       Marker size in the scatter plot. Same convention as
                       pyplot.scatter's *s* parameter.

        *linewidth* : Linewidth for the marker circle. Default: 1.7.

        *edgecolor* : Marker line color. Default: None, which is the color rotation set
                      by this module; gray for a single cluster, default color wheel
                      otherwise.

    Returns
    -------
    fig: plt.figure object.

    axs: List of lists of Axes objects. axs[i][j] is the Axes object corresponding
         to the plot in location i,j.

    """

    # Create the plotting area.
    fig, axs = create_plotting_area(len(x), figsize=figsize, tight=tight, tick_nbins=tick_nbins)

    # Make scatter plots.
    plot_scatter(axs, x, y=y, categories=categories, **kwargs)

    # Annotate the diagonals.
    plot_text(axs, names, diag_fontsize)

    # Make the pie charts.
    plot_pie(axs, x)

    # Format the plot to make the corrgram readable.
    format_plot(axs, tick_fontsize)

    return fig, axs

# Wrapper for corrgram. Used for structured arrays.
def corrgramR(X, fields=None, alias=None, **kwargs):
    """
    This wraps corrgram for more convenient use with structured arrays.

    This takes the structured array `X` and the field names specified in
    `fields` as a list of features to feed into corrgram.


    Parameters
    ----------
    X : A structured ndarray.

    fields : List of strings. These are the names of the fields in `X` that
             will be plotted in the corrgram. If None, then all fields are
             plotted. Default: None.

    alias : List of strings. These are aliases of the field names that will
            be printed in the diagonal elements. If None, the strings in
            `fields` will be printed. Default: None.

    kwargs : Keyword arguments for `corrgram.corrgram` and `scatterR`. The
             notable ones are listed below:

        *y* : 1D ndarray that indicates the label for elements in each member of `x`.
              Default: None.

        *categories* : If `y` is given, then this is the list of categories that will
                       actually be displayed in the scatter plot.
                       Default: None. In this case, display all unique labels in `y`.

        *figsize*: tuple (default=(10,10)).
                   Size of the figure to create.

        *tight*: bool (default=True)
                 Whether to make the scatter plot `tight`.
                 See `axes.locator_params`'s `tight`.

        *diag_fontsize*: int (default=32)
                 Fontsize of the diagonal text labels.

        *tick_nbins*: int (default=5)
                 Adjusts number of ticks in the scatter plot.
                 See `axes.locator_params`'s `nbins`.

        *tick_fontsize*: int (default=14)
                 Fontsize of the axis tick labels.

        *undersample*: float (default=None).
                       The proportion of points to plot in the scatter plot.

        *marker_size*: float (default=None)
                      Marker size in the scatter plot. Same convention as
                      pyplot.scatter's *s* parameter.

    Returns
    -------
    fig: plt.figure object.

    axs: List of lists of Axes objects. axs[i][j] is the Axes object corresponding
         to the plot in location i,j.

    """

    # Unpack the individual columns into an array x.
    x = []
    if not fields:
        fields = X.dtype.names
    for f in fields:
        x.append(X[f])

    # Use the aliases if available
    if not alias:
        alias = fields

    # Call corrgram.
    fig, axs = corrgram(x, alias, **kwargs)

    return fig, axs

if __name__ == '__main__':

    #x1 = np.random.normal(10, 1, 100)
    #x2 = np.random.normal(0, 0.25, 100)
    #x3 = np.random.normal(-10, 2, 100)
    #x4 = np.random.normal(100, 2, 100)
    #y = np.random.choice([0, 1, 2, 3], 100)
    #x = [ x1, x2, x3, x4 ]
    #names = [ r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$' ]
    #fig, axs = corrgram(x, names, y=y, tight=False, tick_nbins=10, tick_fontsize=8,
    #                    undersample=0.5)

    X = np.genfromtxt('datasets/winequality-red.csv', delimiter=';', names=True)
    Y = X['quality']
    fig, axs = corrgramR(X,
                         fields=['fixed_acidity', 'citric_acid', 'alcohol'],
                         alias=['fixed acidity', 'citric acid', 'alcohol'],
                         y=Y, categories=[7, 8],
                         undersample=[0.1, None], tight=True)

    plt.show()
