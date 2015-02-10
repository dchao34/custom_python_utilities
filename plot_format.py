import colorsys
from scipy.constants import golden
import numpy as np
import matplotlib.pyplot as plt

hastie_colors = [ '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#CC79A7', '#D55E00', '#0072B2' ]
hastie_colors1 = [ '#53ACCA', '#FFC466', '#FFAAAA', '#98D68E', '#CE89BE']

ggplot_colors = [ '#47ABAB', '#FFAAAA', '#FFCE6B', '#7C8BB0', '#FFFFAA', '#CE89BE', '#98D68E' ]
ggoplot_dark_colors = [ '#088F8F', '#D46A6A', '#EFA30D', '#314577', '#DADA6D', '#8D3066', '#3F9232' ]

# Get colors evenly distributed around the HSV color wheel.
def get_hsv_colors(ncolors):

    r_0, g_0, b_0 = 225 / 255., 101 / 255., 101 / 255.
    h_0, s_0, v_0 = colorsys.rgb_to_hsv(r_0, g_0, b_0)
    h_list = ((np.linspace(0, 1, ncolors + 1) + h_0) % 1.0).tolist()[:-1]

    color_list = []
    for h in h_list:
        r, g, b = colorsys.hsv_to_rgb(h, s_0, v_0)
        r, g, b = map(lambda x : int(255 * x), [r, g, b])
        r, g, b = map(lambda x : hex(x)[2:], [r, g, b])
        color_list.append('#' + r + g + b)

    return color_list


# Given an Axes object, format it into the style specified in this function.
def format_axes(ax, tick_fontsize=18, spines=False):

    # Autoscale to data
    ax.autoscale(tight=True)

    # Remove unecessary spines
    if not spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    # Adjust ticklabel fontsize
    xlabels = ax.get_xticklabels()
    ylabels = ax.get_yticklabels()
    [ l.update({'fontsize':tick_fontsize}) for l in xlabels ]
    [ l.update({'fontsize':tick_fontsize}) for l in ylabels ]

    return ax


# Wrapper that creates and formats a single figure.
def create_single_figure(figsize=None, tick_fontsize=18, spines=False):

    # Instantiate a figure object
    if not figsize:
        figsize = (6 * golden, 6)
    fig = plt.figure(figsize=figsize)

    # Add an Axes object to the figure and format it
    ax = fig.add_subplot(111)
    ax = format_axes(ax)

    return fig, ax
