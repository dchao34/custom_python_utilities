import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import interpolate
from plot_format import create_single_figure, format_axes

spline_color_wheel = [ '#088F8F', '#D46A6A', '#EFA30D', '#314577', '#DADA6D', '#8D3066', '#3F9232' ]
hist_color_wheel = [ '#47ABAB', '#FFAAAA', '#FFCE6B', '#7C8BB0', '#FFFFAA', '#CE89BE', '#98D68E' ]


# Compute the spline. The course-ness is adjusted using the number of bins used
# to get the "y-coordinate" of the knots.
def spline_hist(x_arr, bins, w_arr):

    # Make a histogram out of x_arr as a reference for the spline knots.
    values, edges = np.histogram(x_arr, bins=bins, weights=w_arr)

    # Compute the knot location of the splines.
    bwidths = np.ediff1d(edges)
    edges_m = np.array([edges[0] - bwidths[0] / 2] +
                       (edges[0:-1] + bwidths / 2).tolist() +
                       [edges[-1] + bwidths[-1] / 2])
    values_m = np.array([0] + values.tolist() + [0])

    # Interpolate the spline.
    spline = interpolate.UnivariateSpline(edges_m, values_m, k=3, ext=1, s=0)

    # Return the spline itself, and the domain where the spline is defined.
    return spline, edges_m[0], edges_m[-1], bwidths[0]


def histR(x,
          show_hist=True, show_spline=False, knots_spline=None,
          hist_colors=None, spline_colors=None,
          lw_spline=None, neval_spline=None,
          xlabel=None, ylabel=None, title=None,
          axislabel_fontsize=20,
          legend=False, legend_names=None,
          legend_loc=None, legend_ncol=None, legend_fontsize=20,
          ax=None, **kwargs):
    """
    This is a wrapper for Axes.hist, but limits/extends its functionality
    to those that are relevant for counting and density visualization.

    This function uses the native functions of Axes.hist, but extends it to
    also plot a spline that gives an approximation to the density estimate.


    Parameters
    ----------
    x : (n,) ndarray or a list of (n,) ndarrays. This is the same as convention
        used in Axes.hist.

    **kwargs : Same as those in Axes.hist. Notable ones are:

        *weights* : Weights for the data points in x.
                    Default: None.

        *bins* : Number of bins for the histogram.
                 Default: 10.

        *normed* : Whether to normalize the histogram. If True, the spline
                   will also be normalized.
                   Default: False.

        *stacked* : Whether to stack the histogram. If True, the spline
                    will trace the overall result.
                    Default: False.

    show_hist: Bool. Whether to display the histogram. Default: True.

    show_spline: Bool. Whether to display the spline. When show_hist=True,
                 the spline hugs the histogram outline. When show_hist=False, the
                 spline integrates to the weighted counts of the input data.
                 Default: False.

    knots_spline: Float or a list of floats. This adjust the coarseness of the spline
                  interpolation. It is the same as the number of bins used to determine
                  the knot locations in the spline computation. Specify a list if you
                  want to adjust each spline component separately.
                  Default: 10

    lw_spline : Linewidth of the splines. Default: 2.0.

    hist_colors : List of strings. This is the color rotation used for the histograms.

    spline_colors : List of strings. This is the color rotation used for the splines.

    ax : This is the axes to which the drawing is directed.
         Default: None, which converts to pyplot.gca().

    neval_spline: Int. The number of points to evaluate the spline for display.
                  Default: 1000

    xlabel : X axis label.

    ylabel : Y axis label.

    title : Plot title.

    axislabel_fontsize: Fontsize of axis tick label. Default: 20.

    legend : Whether to include a legend.
             Default: False.

    legend_names : List of strings. Element i corresonds to component i of the
                   histogram. Default: None, which gets converted to
                   numbers, one for each category.

    legend_loc : Legend location; same as *loc* keyword argument for pyplot.legend().
                 Default: None, which is 'best'.

    legend_ncol : Number of columns in the legend; same as *ncol* keyword argument for
                  pyplot.legend(). Default: 1.

    legend_fontsize : Legend fontsize. Default: 20.

    Returns
    -------
    This function does not return anything.
    (In contrast to the original `pyplot.scatter()` function)

    """

    # Initial configurations
    # ----------------------

    # Get Axes object and add labels.
    if ax is None: ax = format_axes(plt.gca())
    if xlabel: ax.set_xlabel(xlabel, fontsize=axislabel_fontsize)
    if ylabel: ax.set_ylabel(ylabel, fontsize=axislabel_fontsize)
    if title: ax.set_title(title, fontsize=axislabel_fontsize)

    # Always handle input data as a list of ndarrays
    if isinstance(x, np.ndarray): x = [ x ]

    # Need to know the type of histogram the user intends to plot in order to
    # determine the details of the plot layout.
    stacked, normed = False, False
    if 'stacked' in kwargs: stacked = kwargs['stacked']
    if 'normed' in kwargs: normed = kwargs['normed']

    # Default to a known number of bins for the histogram
    bins = 10
    if 'bins' in kwargs: bins = kwargs['bins']

    # Plot histogram
    # --------------

    # Binwidth of the histogram. Spline plotting needs this parameter to
    # display the matching with the histogram correctly.
    bwidth_hist = 1

    # Need the max histogram count to adjust the Axes's ylimits.
    hist_ymax = 0

    hist_legend_handles = []
    if show_hist:

        # Configure histogram patch colors. If no colors were explicitly given,
        # then default to the preconfigured color wheel.
        if hist_colors is None:
            hist_colors = [ hist_color_wheel[ i % len(hist_color_wheel) ]
                            for i in range(len(x)) ]
        kwargs['color'] = hist_colors

        # Adjust the patch widths to occupy the entire bin. There's still going to
        # be spacing between the patches, since the patch linewidth is non-zero.
        if 'rwidth' not in kwargs: kwargs['rwidth'] = 1

        # Plot the histogram. Just forward all kwargs to `Axes.hist` for plotting.
        hist_counts, bin_edges, hist_patches = ax.hist(x, **kwargs)
        bwidth_hist = bin_edges[1] - bin_edges[0]

        # Adjust the line color of the histogram patches.
        if isinstance(hist_counts[0], np.ndarray):
            for i, patch_list in enumerate(hist_patches):
                for patch in patch_list: patch.set_ec('white')
                hist_legend_handles.append(hist_patches[i][0])
        else:
            for patch in hist_patches: patch.set_ec('white')
            hist_legend_handles.append(hist_patches[0])

        # Get maximum histogram count
        if isinstance(hist_counts[0], np.ndarray):
            hist_ymax = max([ np.max(hcts) for hcts in hist_counts ])
        else:
            hist_ymax = np.max(hist_counts)


    # Plot Spline
    # -----------

    # Need the max spline y-coordinate to adjust the Axes's ylimits.
    spline_ymax = 0

    spline_legend_handles = []
    if show_spline:

        # General configuration for splines.
        # ----------------------------------

        # Handle the weight parameter as either None or a list of ndarrays.
        weights = None
        if 'weights' in kwargs: weights = kwargs['weights']
        if isinstance(weights, np.ndarray): weights = [ weights ]

        # General spline display configurations.
        if neval_spline is None: neval_spline = 1000
        if lw_spline is None: lw_spline = 2.0

        # For stacked histograms, the spline should interpolate the stacked
        # result and be agnostic to the contituents.
        if stacked:

            # Concatenate all inputs and weights
            x_all, w_all = np.concatenate(x), weights
            if weights is not None: w_all = np.concatenate(weights)

            # Configure the number of knots that the spline should have. If a
            # list is given, then take the minimum value.
            if knots_spline is None: knots_spline = 10
            if isinstance(knots_spline, (list, tuple)):
                knots_spline = min(knots_spline)

            # Configure spline colors. Default to the preconfigured single
            # color, unless a list of length 1 is given.
            if spline_colors is None: spline_colors = spline_color_wheel
            if isinstance(spline_colors, (list, tuple)):
                if len(spline_colors) > 1: spline_colors = spline_color_wheel

            # Compute the spline. Returns the spline, the range for which it is
            # defined, and the binwidth used to construct the knots.
            spline, xmin_sp, xmax_sp, bwidth_sp = spline_hist(x_all, knots_spline, w_all)

            # Compute the normalization before plotting
            # -----------------------------------------

            # If the histogram is shown, need to adjust the normalization of
            # the spline so that it actually hugs the histogram.
            normalization = bwidth_sp / bwidth_hist

            # In the normed case, simply divide by the integral.
            if normed:
                normalization = spline.integral(xmin_sp, xmax_sp)

            # The only case left is the un-normed case when the histogram is
            # absent. In this case, arrange so that the integral of the spline
            # gives the total weighted counts.
            else:
                if not show_hist:
                    if w_all is None:
                        normalization = spline.integral(xmin_sp, xmax_sp) / x_all.shape[0]
                    else:
                        normalization = spline.integral(xmin_sp, xmax_sp) / np.sum(w_all)

            # Plot the spline
            x_sp = np.linspace(xmin_sp, xmax_sp, neval_spline)
            y_sp = spline(x_sp) / normalization
            lines = ax.plot(x_sp, y_sp,
                            lw=lw_spline, color=spline_colors[0] )
            spline_legend_handles.append(lines[0])
            spline_ymax = np.max(y_sp)

        # For non-stacked histograms, the spline should interpolate each
        # constituent separately.
        else:

            # Configure spline colors. Default to preconfigured color wheel
            if spline_colors is None: spline_colors = spline_color_wheel

            # Make a list out of the weights for looping.
            if weights is None: weights = [ None ] * len(x)

            # Make a list out of the spline knots for looping
            if knots_spline is None: knots_spline = 10
            if not isinstance(knots_spline, (list, tuple)):
                knots_spline = [ knots_spline ] * len(x)

            # Consider each constituent iteratively.
            for i, (x_i, knots_i, w_i) in enumerate(zip(x, knots_spline, weights)):

                # Compute the spline.
                spline, xmin_sp, xmax_sp, bwidth_sp= spline_hist(x_i, knots_i, w_i)

                # Compute the normalization before plotting. Same logic as the
                # stacked case, but this time perform normalization for
                # individual constituents.
                normalization = bwidth_sp / bwidth_hist
                if normed:
                    normalization = spline.integral(xmin_sp, xmax_sp)
                else:
                    if not show_hist:
                        if w_i is None:
                            normalization = spline.integral(xmin_sp, xmax_sp) / x_i.shape[0]
                        else:
                            normalization = spline.integral(xmin_sp, xmax_sp) / np.sum(w_i)

                # Plot the spline
                x_sp = np.linspace(xmin_sp, xmax_sp, neval_spline)
                y_sp = spline(x_sp) / normalization
                lines = ax.plot(x_sp, y_sp,
                                lw=lw_spline, color=spline_colors[i % len(spline_colors)] )
                spline_legend_handles.append(lines[0])
                spline_ymax = max(np.max(y_sp), spline_ymax)

    ax.set_ylim(0, 1.05 * max(hist_ymax, spline_ymax))

    # Build legend
    if legend:

        # General configurations
        if legend_loc is None: legend_loc = 'best'
        if legend_ncol is None: legend_ncol = 1
        if legend_names is None: legend_names = map(str, range(len(x)))


        # When show_hist is on, just use the histogram handles. The stacked
        # spline will also not be present in the legend.
        legend_handles = []
        if show_hist:
            legend_handles = hist_legend_handles

        # When show_hist is off, use the spline handles. If stacked=True, then
        # we need to coerce the legend to a single entry.
        else:
            legend_handles = spline_legend_handles
            if stacked:
                legend_names = [ 'stacked spline' ]

        ax.legend(legend_handles, legend_names,
                    loc=legend_loc, ncol=legend_ncol,
                    fontsize=legend_fontsize)

    return

if __name__ == "__main__":

    data1 = np.random.normal(-5, 4, 1000)
    data2 = np.random.normal(5, 4, 1000)
    weight1 = np.ones(1000)
    weight2 = 5.0 * np.ones(1000)

    #fig = plt.figure(figsize=(20,10))
    #ax1 = format_axes(fig.add_subplot(221))
    #histR([data1, data2], ax=ax1,
    #      weights =[weight1, weight2],
    #      show_spline=True, show_hist=True,
    #      bins=30, stacked=False, normed=False)
    #ax2 = format_axes(fig.add_subplot(222))
    #histR([data1, data2], ax=ax2,
    #      weights =[weight1, weight2],
    #      show_spline=True, show_hist=True,
    #      bins=30, stacked=False, normed=True)
    #ax3 = format_axes(fig.add_subplot(223))
    #histR([data1, data2], ax=ax3,
    #      weights =[weight1, weight2],
    #      show_spline=True, show_hist=True,
    #      bins=30, stacked=True, normed=False)
    #ax4 = format_axes(fig.add_subplot(224))
    #histR([data1, data2], ax=ax4,
    #      weights =[weight1, weight2],
    #      show_spline=True, show_hist=True,
    #      bins=30, stacked=True, normed=True)

    fig, ax = create_single_figure()
    histR([data1, data2], weights=[weight1, weight2],
          knots_spline=10,
          show_hist=True, show_spline=True,
          bins=20, stacked=False, normed=False,
          xlabel='Feature', ylabel='Counts', title='Stacked',
          legend=True, legend_names=['label1', 'label2'])

    #histR([data1, data2], weights=None, bins=10, show_spline=True,
    #      stacked=True, normed=True)
    #histR([data1, data2], bins=10, show_spline=True, spline_colors=['blue'],
    #      stacked=True, normed=False)
    #histR(data1, weights=weight2, lw=None)
    #histR(data1, weights=None, lw=None)
    #histR(data1, lw=None)

    plt.show()

