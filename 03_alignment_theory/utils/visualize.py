from itertools import product

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils.analysis import predecessor2path

edge_delta = {'-': (.9, 0.), '|': (0., -.9), '\\': (.9, -.9)}

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, family='Monospace')
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, family='Monospace')

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:d}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_alignment(ax, S, X, Y, P=None, best=None, cmap='plasma', 
                   annotate=True, textcolors=('white', 'black'), threshold=None, 
                   pathcolor='cyan', pathalpha=.25, pathwidth=7):
    im, cbar = heatmap(S, list('-' + X), list('-' + Y), ax=ax, cmap=cmap)
    texts = []
    if annotate:
        texts = annotate_heatmap(im, textcolors=textcolors, threshold=threshold)
    if P is not None:
        path = predecessor2path(P, best)
        ax.plot(path[1], path[0], linewidth=pathwidth, color=pathcolor, alpha=pathalpha)
    return im, cbar, texts


def draw_arrow(ax, x, y, d, head_width=0.1, head_length=0.3, color='black', **kwargs):
    dx, dy = edge_delta[d]
    ax.arrow(x, y, dx, dy, length_includes_head=True, 
             head_width=head_width, head_length=head_length, 
             facecolor=color, edgecolor=color,
             **kwargs)

def plot_path_graph(ax, S1, S2):
    m, n = len(S1), len(S2)
    y, x = zip(*product(range(m + 1), range(n + 1)))
    
    # plot nodes
    ax.plot(x, y, 'o', markerfacecolor='white', markeredgecolor='black', markersize=10)
    # plot edges
    for i in range(n-1, -1, -1):  # reverse order to plot from top to bottom
        for j in range(m, 0, -1):
            draw_arrow(ax, i, j, '-')
            draw_arrow(ax, i, j, '|')
            draw_arrow(ax, i, j, '\\')        
    for i in range(n-1, -1, -1):  # last row
        draw_arrow(ax, i, 0, '-')
    for j in range(m, 0, -1): # last column
        draw_arrow(ax, n, j, '|')

    ax.axes.set_aspect('equal')
    ax.tick_params(length=0, labelbottom=False, labeltop=True)
    ax.set_xticks(ticks=np.arange(n) + .5, labels=S2)
    ax.set_yticks(ticks=np.arange(m) + .5, labels=S1[::-1])
    ax.set_frame_on(False)

def plot_graph_alignment(ax, S1, S2, cigar):
    m, n = len(S1), len(S2)
    i, j = 0, m
    for a in cigar:
        if a == 'M':
            draw_arrow(ax, i, j, '\\', color='red', width=0.01)
            i, j = i+1, j-1
        elif a == 'D':
            draw_arrow(ax, i, j, '-', color='red', width=0.01)
            i += 1
        elif a == 'I':
            draw_arrow(ax, i, j, '|', color='red', width=0.01)
            j -= 1
        else:
            raise ValueError('Invalid CIGAR string')
        if i > n or j < 0:
            raise ValueError('Invalid CIGAR string')

            