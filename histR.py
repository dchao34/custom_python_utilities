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


def histR(x, show_hist=True, show_spline=False,
          knots_spline=None, neval_spline=None,
          hist_colors=None, spline_colors=None,
          lw_spline=None,
          ax=None, **kwargs):
    """
    hist_colors : List of strings.
    """

    # Initial configurations
    # ----------------------

    # If no Axes is given, use plt.gca() and give it a standard look
    if ax is None: ax = format_axes(plt.gca())

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
            for patch_list in hist_patches:
                for patch in patch_list: patch.set_ec('white')
        else:
            for patch in hist_patches: patch.set_ec('white')


    # Plot Spline
    # -----------

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
            ax.plot(x_sp, spline(x_sp) / normalization,
                    lw=lw_spline, color=spline_colors[0] )

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
                ax.plot(x_sp, spline(x_sp) / normalization,
                        lw=lw_spline, color=spline_colors[i % len(spline_colors)] )

    ax.set_ylim(0)

    return

if __name__ == "__main__":

    data1 = np.random.normal(-5, 4, 1000)
    data2 = np.random.normal(5, 4, 1000)
    weight1 = np.ones(1000)
    weight2 = 5.0 * np.ones(1000)

    fig = plt.figure(figsize=(20,10))
    ax1 = format_axes(fig.add_subplot(221))
    histR([data1, data2], ax=ax1,
          weights =[weight1, weight2],
          show_spline=True, show_hist=True,
          bins=30, stacked=False, normed=False)
    ax2 = format_axes(fig.add_subplot(222))
    histR([data1, data2], ax=ax2,
          weights =[weight1, weight2],
          show_spline=True, show_hist=True,
          bins=30, stacked=False, normed=True)
    ax3 = format_axes(fig.add_subplot(223))
    histR([data1, data2], ax=ax3,
          weights =[weight1, weight2],
          show_spline=True, show_hist=True,
          bins=30, stacked=True, normed=False)
    ax4 = format_axes(fig.add_subplot(224))
    histR([data1, data2], ax=ax4,
          weights =[weight1, weight2],
          show_spline=True, show_hist=True,
          bins=30, stacked=True, normed=True)

    #fig, ax = create_single_figure()
    #histR([data1, data2], weights=[weight1, weight2],
    #      knots_spline=10, show_spline=True,
    #      bins=20, stacked=True, normed=False)
    #histR([data1, data2], weights=None, bins=10, show_spline=True,
    #      stacked=True, normed=True)
    #histR([data1, data2], bins=10, show_spline=True, spline_colors=['blue'],
    #      stacked=True, normed=False)
    #histR(data1, weights=weight2, lw=None)
    #histR(data1, weights=None, lw=None)
    #histR(data1, lw=None)

    plt.show()

