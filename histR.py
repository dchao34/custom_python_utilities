import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import constants
from scipy import interpolate

hist_color = [ '#47ABAB', '#FFAAAA', '#FFCE6B', '#7C8BB0', '#FFFFAA', '#CE89BE', '#98D68E' ]
spline_color = [ '#088F8F', '#D46A6A', '#EFA30D', '#314577', '#DADA6D', '#8D3066', '#3F9232' ]

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

    return spline, edges_m[0], edges_m[-1], bwidths[0]

def histR(arr, weights=None, ax=None,
          bins=None, spline_bins=None, normed=False,
          show_hist=True, show_spline=True,
          xlabel=None, ylabel=None, title=None,
          axislabel_fontsize=20,
          hist_lw=0.5, spline_lw=4, spline_style='-',
          spline_nevalpts=1000, **kwargs):
    """
    Plot a histogram with extended functionality to interpolate and normalize.

    This function extends the `hist()` function built into matplotlib.
    It computes and plots the histogram of *arr*, but can optionally
    compute and plot an interpolated spline. One can request the plots
    to be normalized to give an approximation of the density function.
    (It is of course not a real density estimate, but gives a similar
    form.)

    The splines are computed using histogram values. You can tune the
    distance between the nots by changing the bin number. Note that the
    number of bins for the spline interpolation need not be the same as
    for the histogram.

    You can also input multiple data for *arr*. You specify this the
    same way as you would have for matplotlib's `hist()` function.
    For example, *arr* = [ arr1, arr2, arr3 ].

    Parameters
    ----------
    arr : (n,) array or list of (n,) arrays
          Input values, this takes either a single array or a list of
          arrays which are not required to be of the same length.

    weights : (n,) array or list of (n,) arrays
              Input weights for `arr`. If a list is given, it must be
              the same length as `arr`.

    bins : Integer, optional. Default: Freedman-diaconis.
           Number of bins for the histogram.

    spline_bins : Integer or a list of integers, optional. Default: Freedman-diaconis.
                  Number of bins for the spline interpolation.
                  If a list is provided, it should have the same length as `arr`.

    normed : Normalize the results before plotting. Default: False.

    show_hist : Plot the histogram. Default: True.

    Other formal keyword arguments are for plot formatting.

    kwargs are the same as those for the original `hist()` function.

    Returns
    -------
    This function does not return anything.
    (In contrast to the original `hist()` function)

    """

    fig = None
    if ax is None:
        fig = plt.figure(figsize=(6 * constants.golden, 6))
        ax = fig.add_subplot(111)
    format_axes(ax)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=axislabel_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=axislabel_fontsize)
    if title:
        ax.set_title(title, fontsize=axislabel_fontsize)

    if not isinstance(arr, list):
        arr = [ arr ]

    if weights is None:
        weights = [ None ] * len(arr)
    elif not isinstance(weights, list):
        weights = [ weights ]

    if spline_bins is None:
        spline_bins = [ None ] * len(arr)
    elif not isinstance(spline_bins, list):
        spline_bins = [ spline_bins ]

    for i in range(len(arr)):

        if bins is None:
            bins = freedman_diaconis(arr[i])
        if spline_bins[i] is None:
            spline_bins[i] = freedman_diaconis(arr[i])

        hist_bwidth = 1
        if show_hist:
            values, edges, patches = ax.hist(arr[i], bins=bins,
                                             weights=weights[i], normed=normed,
                                             lw=hist_lw,
                                             color=hist_color[i % len(hist_color)],
                                             **kwargs)
            hist_bwidth = edges[1] - edges[0]
            for p in patches: p.set_ec('white')

        if show_spline:
            spline, spline_xmin, spline_xmax, spline_bwidth = spline_hist(arr[i], spline_bins[i], weights[i])
            spline_x = np.linspace(spline_xmin, spline_xmax, spline_nevalpts)

            if normed:
                normalization = spline.integral(spline_xmin, spline_xmax)
            else:
                normalization = spline_bwidth / hist_bwidth
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
    histR([data1, data2], ax=ax,
          bins=20, spline_bins=[10, 10],
          xlabel=r"Feature (Units)", ylabel='Counts')
    plt.show()

