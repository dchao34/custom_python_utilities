import numpy as np
import matplotlib.pyplot as plt
from plot_format import format_axes

color = [ '#53ACCA', '#FFC466', '#FFAAAA', '#98D68E', '#CE89BE']

def format_axes(ax, tick_fontsize=18):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    return ax

def scatterR(x, y,
             ax=None,
             s=80, linewidth=1.7,
             facecolor='none', edgecolor='#5588BE',
             xlabel=None, ylabel=None, title=None,
             axislabel_fontsize=20):

    if not isinstance(x, (list, tuple)):
        x, y = [ x ], [ y ]

    if not ax:
        ax = plt.gca()
    format_axes(ax)
    print len(x)
    for i in range(len(x)):
        ax.scatter(x[i], y[i],
                   s=s, linewidth=linewidth,
                   facecolor=facecolor,
                   edgecolor=color[i % len(color)])

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=axislabel_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=axislabel_fontsize)
    if title:
        ax.set_title(title, fontsize=axislabel_fontsize)

if __name__ == "__main__":

    (x1, y1) = np.random.normal(3, 1, 500), np.random.normal(3, 2, 500)
    (x2, y2) = np.random.normal(-3, 2, 500), np.random.normal(-3, 1, 500)
    (x3, y3) = np.random.normal(-3, 2, 500), np.random.normal(3, 1, 500)
    (x4, y4) = np.random.normal(6, 2, 500), np.random.normal(3, 1, 500)
    (x5, y5) = np.random.normal(6, 2, 500), np.random.normal(-3, 1, 500)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    scatterR([x1, x2, x3, x4, x5],
             [y1, y2, y3, y4, y5],
             ax=ax, xlabel='feature 1', ylabel='feature 2')
    plt.show()
