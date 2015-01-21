import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import constants
from scipy import interpolate
from sklearn.neighbors import KernelDensity

hist_color = [ '#00CCCC', '#FFCCCC', '#FFFF99' ]
spline_color = [ '#009999', '#FF9999', '#FFFF66' ]

def format_axes(ax, tick_fontsize=18):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    return ax

def freedman_diaconis(arr):
    iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
    bins = 2 * iqr * scipy.special.cbrt(len(arr))
    return bins

def spline_hist(arr, bins, weights):

    values, edges = np.histogram(arr, bins=bins, weights=weights)
    bwidths = np.ediff1d(edges)
    edges_m = np.array([edges[0] - bwidths[0] / 2] +
                       (edges[0:-1] + bwidths / 2).tolist() +
                       [edges[-1] + bwidths[-1] / 2])
    values_m = np.array([0] + values.tolist() + [0])

    spline = interpolate.UnivariateSpline(edges_m, values_m, k=3, ext=1, s=0)

    return spline, edges_m[0], edges_m[-1]

def histR(arr, weights=None, ax=None,
          bins=None, show_hist=True,
          spline_bins=None, normed=False,
          xlabel=None, ylabel=None, title=None,
          axislabel_fontsize=20,
          hist_lw=0.5, spline_lw=4, spline_style='-',
          spline_nevalpts=1000, **kwargs):
    """
    Plot a histogram with extended functionality.

    Compute and draw the histogram of *x*. The return value is a
    tuple (*n*, *bins*, *patches*) or ([*n0*, *n1*, ...], *bins*,
    [*patches0*, *patches1*,...]) if the input contains multiple
    data.

    Multiple data can be provided via *x* as a list of datasets
    of potentially different length ([*x0*, *x1*, ...]), or as
    a 2-D ndarray in which each column is a dataset.  Note that
    the ndarray form is transposed relative to the list form.

    Masked arrays are not supported at present.

    Parameters
    ----------
    x : (n,) array or sequence of (n,) arrays
        Input values, this takes either a single array or a sequency of
        arrays which are not required to be of the same length

    bins : integer or array_like, optional, default: 10
        If an integer is given, `bins + 1` bin edges are returned,
        consistently with :func:`numpy.histogram` for numpy version >=
        1.3.

        Unequally spaced bins are supported if `bins` is a sequence.
    """

    if ax is None:
        ax = plt.gca()
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=axislabel_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=axislabel_fontsize)
    if title:
        ax.set_title(title, fontsize=axislabel_fontsize)

    if not isinstance(arr, list):
        arr = [ arr ]

    if bins is None:
        bins = [ None ] * len(arr)
    elif not isinstance(bins, list):
        bins = [ bins ]

    if spline_bins is None:
        spline_bins = [ None ] * len(arr)
    elif not isinstance(spline_bins, list):
        spline_bins = [ spline_bins ]
    if not normed:
        spline_bins = bins

    for i in range(len(arr)):

        if bins[i] is None:
            bins[i] = freedman_diaconis(arr[i])
        if spline_bins[i] is None:
            spline_bins[i] = freedman_diaconis(arr[i])

        if show_hist:
            values, edges, patches = ax.hist(arr[i], bins=bins[i],
                                             weights=weights, normed=normed,
                                             lw=hist_lw,
                                             color=hist_color[i % len(hist_color)],
                                             **kwargs)
            for p in patches: p.set_ec('white')

        spline, spline_xmin, spline_xmax = spline_hist(arr[i], spline_bins[i], weights)
        spline_x = np.linspace(spline_xmin, spline_xmax, spline_nevalpts)
        normalization = 1
        if normed:
            normalization = spline.integral(spline_xmin, spline_xmax)
        ax.plot(spline_x, spline(spline_x) / normalization,
                spline_style, lw=spline_lw,
                color=spline_color[i % len(spline_color)])

    ax.set_ylim(0)

    return

if __name__ == "__main__":

    data1 = np.random.normal(-5, 4, 1000)
    data2 = np.random.normal(5, 4, 1000)
    fig = plt.figure(figsize=(6 * constants.golden, 6))
    ax = fig.add_subplot(111)
    format_axes(ax)
    histR([data1, data2], ax=ax,
          bins=[10, 20], show_hist=True,
          spline_bins=[10, 20], normed=False,
          xlabel=r"$E_{extra}$ (GeV)", ylabel='Counts')
    plt.show()

