import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from plot_format import format_axes
from scatterR import scatterR

def create_plotting_area(M, figsize=None, tight=None, nbins=None):
    if not figsize:
        figsize=(10,10)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(M, M, wspace=0.0, hspace=0.0)
    axs = []
    for i in range(M):
        row = []
        for j in range(M):
            ax = plt.subplot(gs[i,j])

            # Adjust the number of axis ticks.
            # Don't tamper with the diagonal plot areas.
            if i != j:
                ax.locator_params(tight=tight, nbins=nbins)
            row.append(ax)
        axs.append(row)
    return fig, axs

def plot_scatter(axs, x, undersample=None, s=None):

    scatterR_kwargs = {}
    if s:
        scatterR_kwargs['s'] = s

    it = itertools.combinations(range(len(x)), 2)
    while True:
        try:
            i, j = it.next()
            x1, x2 = x[i], x[j]
            if undersample:
                m = np.zeros(len(x1), dtype=bool)
                pts = int(len(x1) * undersample)
                m[:pts] = True
                np.random.shuffle(m)
                x1, x2 = x1[m], x2[m]
            scatterR(x2, x1, ax=axs[i][j], edgecolor=['gray'], **scatterR_kwargs)
        except StopIteration:
            break

def plot_text(axs, names, name_fontsize):
    if not names:
        return
    for i, name in enumerate(names):
        axs[i][i].text(0.5, 0.5, name, fontsize=name_fontsize, ha='center', va='center')

def plot_pie(axs, x):

    it = itertools.combinations(range(len(x)), 2)
    while True:
        try:
            i, j = it.next()
            corr_coef = stats.pearsonr(x[i], x[j])[0]
            sizes, colors = [], []
            if corr_coef > 0:
                sizes = [ 1 - corr_coef, corr_coef ]
                colors = [ '#FFFFFF', '#D46A6A' ]
            else:
                sizes = [ - corr_coef, 1 + corr_coef ]
                colors = [ '#4E648E', '#FFFFFF' ]

            pie_wedges, pie_text = axs[j][i].pie(sizes, startangle=90, colors=colors)
            for wedge in pie_wedges: wedge.set_ec('gray')

        except StopIteration:
            break

def format_plot(axs, tick_labelsize=None):

    M = len(axs)
    it = itertools.product(range(M), range(M))
    while True:
        try:
            i, j = it.next()
            ax = axs[i][j]
            if tick_labelsize:
                ax.tick_params(labelsize=tick_labelsize)

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

def corrgram(x, names,
             undersample=None,
             marker_size=None,
             tight=True, nbins=5,
             figsize=None,
             name_fontsize=32, tick_labelsize=14):
    fig, axs = create_plotting_area(len(x), figsize=figsize, tight=tight, nbins=nbins)
    plot_scatter(axs, x, undersample, s=marker_size)
    plot_text(axs, names, name_fontsize)
    plot_pie(axs, x)
    format_plot(axs, tick_labelsize)
    return fig, axs

def corrgramR(X, fields=None, names=None, **kwargs):
    x = []
    if not fields:
        fields = X.dtype.names
    for f in fields:
        x.append(X[f])

    if not names:
        names = fields
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
                         undersample=0.1, tight=True)
    plt.show()
