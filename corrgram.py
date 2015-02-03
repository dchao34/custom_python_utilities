import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from plot_format import format_axes
from scatterR import scatterR

# Create the plotting area. Return the figure object and a list of list of Axes
# objects. We can access the (i,j)th Axes object in axs[i][j].
def create_plotting_area(M, figsize=None, tight=None, nbins=None):
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
                ax.locator_params(tight=tight, nbins=nbins)

            row.append(ax)
        axs.append(row)
    return fig, axs

# Make scatter plots in the upper triangular entries.
def plot_scatter(axs, x, undersample=None, s=None):

    # Assemble keyword arguments to pass to scatterR.
    scatterR_kwargs = {}
    if s:
        scatterR_kwargs['s'] = s

    # To undersample, randomly select entries to include in the scatter
    # plot; the entries that are plotted are the same for every feature
    # column. Store the entries that will be plotted as an index array to use
    # later.
    m = np.ones(len(x[0]), dtype=bool)
    if undersample:
        m = np.zeros(len(x[0]), dtype=bool)
        pts = int(len(x[0]) * undersample)
        m[:pts] = True
        np.random.shuffle(m)

    it = itertools.combinations(range(len(x)), 2)
    while True:
        try:
            i, j = it.next()

            # Get the ndarrays that we would like to plot.
            x1, x2 = x[i], x[j]
            x1, x2 = x1[m], x2[m]

            # Plot scatter plot using scatterR. Note that x2 comes before x1.
            scatterR(x2, x1, ax=axs[i][j], edgecolor=['gray'], **scatterR_kwargs)
        except StopIteration:
            break

# Annotate the diagonal plots. Simply place the feature name in the center.
def plot_text(axs, names, name_fontsize):
    if not names:
        return
    for i, name in enumerate(names):
        axs[i][i].text(0.5, 0.5, name, fontsize=name_fontsize, ha='center', va='center')

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
def format_plot(axs, tick_labelsize=None):

    M = len(axs)
    it = itertools.product(range(M), range(M))
    while True:
        try:
            i, j = it.next()
            ax = axs[i][j]

            # Font size for the tick labels.
            if tick_labelsize:
                ax.tick_params(labelsize=tick_labelsize)

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
             undersample=None,
             marker_size=None,
             tight=True, nbins=5,
             figsize=None,
             name_fontsize=32, tick_labelsize=14):
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


    Parameters
    ----------
    x : List of 1 dimension ndarray's of equal length.

    names : List of strings. These are aliases of the field names that will
            be printed in the diagonal elements. If None, the strings in
            `fields` will be printed. Default: None.

    kwargs : Keyword arguments for `corrgram.corrgram`. These include:
        *figsize*: tuple (default=(10,10)).
                   Size of the figure to create.
        *undersample*: float (default=None).
                       The proportion of points to plot in the scatter plot.
        *marker_size*: float (default=None)
                      Marker size in the scatter plot. Same convention as
                      pyplot.scatter's *s* parameter.
        *tight*: bool (default=True)
                 Whether to make the scatter plot `tight`. See `axes.locator_params`.
        *nbins*: int (default=5)
                 Adjusts number of ticks in the scatter plot. See `axes.locator_params`.
        *name_fontsize*: int (default=32)
                 Fontsize of the diagonal text labels.
        *tick_labelsize*: int (default=14)
                 Fontsize of the axis tick labels.

    Returns
    -------
    fig: plt.figure object.

    axs: List of lists of Axes objects. axs[i][j] is the Axes object corresponding
         to the plot in location i,j.

    """

    # Create the plotting area.
    fig, axs = create_plotting_area(len(x), figsize=figsize, tight=tight, nbins=nbins)

    # Make scatter plots.
    plot_scatter(axs, x, undersample, s=marker_size)

    # Annotate the diagonals.
    plot_text(axs, names, name_fontsize)

    # Make the pie charts.
    plot_pie(axs, x)

    # Format the plot to make the corrgram readable.
    format_plot(axs, tick_labelsize)

    return fig, axs

# Wrapper for corrgram. Used for structured arrays.
def corrgramR(X, fields=None, names=None, **kwargs):
    """
    Plot a correlogram of a structured array. This was inspired by a
    similar function in R. This is really a wrapper for `corrgram.corrgram`.

    This takes the structured array `X` and the field names specified in
    `fields` to make a grid of plots. The upper triagular elements are
    scatterplots between each pair of fields. The diagonal specifies the
    field names that are being plotted. The lower triagular elements are
    pie charts that indicates the pearson correlation coefficient.


    Parameters
    ----------
    X : A structured ndarray.

    fields : List of strings. These are the names of the fields in `X` that
             will be plotted in the corrgram. If None, then all fields are
             plotted. Default: None.

    names : List of strings. These are aliases of the field names that will
            be printed in the diagonal elements. If None, the strings in
            `fields` will be printed. Default: None.

    kwargs : Keyword arguments for `corrgram.corrgram`. These include:
        *figsize*: tuple (default=(10,10)).
                   Size of the figure to create.
        *undersample*: float (default=None).
                       The proportion of points to plot in the scatter plot.
        *markersize*: float (default=None)
                      Marker size in the scatter plot. Same convention as
                      pyplot.scatter's *s* parameter.
        *tight*: bool (default=True)
                 Whether to make the scatter plot `tight`. See `axes.locator_params`.
        *nbins*: int (default=5)
                 Adjusts number of ticks in the scatter plot. See `axes.locator_params`.
        *name_fontsize*: int (default=32)
                 Fontsize of the diagonal text labels.
        *tick_labelsize*: int (default=14)
                 Fontsize of the axis tick labels.

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
    if not names:
        names = fields

    # Call corrgram.
    fig, axs = corrgram(x, names, **kwargs)

    return fig, axs

if __name__ == '__main__':

    #(x1, y1) = np.random.normal(3, 1, 500), np.random.normal(3, 4, 500)
    #x1 = np.random.normal(10, 1, 100)
    #x2 = np.random.normal(0, 0.25, 100)
    #x3 = np.random.normal(-10, 2, 100)
    #x4 = np.random.normal(100, 2, 100)
    #x = [ x1, x2, x3, x4 ]
    #names = [ r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$' ]
    #fig, axs = corrgram(x, names, undersample=0.7, tight=False, nbins=7, marker_size=10)
    #plt.show()

    X = np.genfromtxt('datasets/winequality-red.csv', delimiter=';', names=True)
    #fig, axs = corrgramR(X, ['fixed_acidity', 'citric_acid', 'alcohol'], undersample=0.1, tight=True)
    #fig, axs = corrgramR(X, figsize=(20, 10), undersample=0.1,
    #                     tight=True, name_fontsize=10, tick_labelsize=10,
    #                     marker_size=10)
    fig, axs = corrgramR(X,
                         fields=['fixed_acidity', 'citric_acid', 'alcohol'],
                         names=['fixed acidity', 'citric acid', 'alcohol'],
                         undersample=0.01, tight=True)
    plt.show()
