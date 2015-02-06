from scipy.constants import golden
import matplotlib.pyplot as plt

hastie_colors = [ '#53ACCA', '#FFC466', '#FFAAAA', '#98D68E', '#CE89BE']
ggplot_colors = [ '#47ABAB', '#FFAAAA', '#FFCE6B', '#7C8BB0', '#FFFFAA', '#CE89BE', '#98D68E' ]
ggoplot_dark_colors = [ '#088F8F', '#D46A6A', '#EFA30D', '#314577', '#DADA6D', '#8D3066', '#3F9232' ]

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
