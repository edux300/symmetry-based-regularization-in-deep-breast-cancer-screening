
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

def show_img(img, cmap=None):
    fig, ax = plt.subplots()  # an empty figure with no axes
    if len(img.shape) == 2 and cmap is None:
        cmap = "gray"
    ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    return fig

def histogram(array, n_bins=None, range=(0, 255)):
    fig, ax = plt.subplots()  # an empty figure with no axes
    n_bins = 20 if n_bins is None else n_bins
    _, _, _ = ax.hist(array.reshape((-1)), n_bins, range, density=True, facecolor='blue', alpha=0.5)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Probability")
    return fig

def save_fig(fig, file, close=True):
    fig.savefig(file, bbox_inches = 'tight', pad_inches = 0)
    if close:
        plt.close(fig)

def running_mean(x, n):
    x = np.pad(x, (n+1, n), "edge")
    cumsum = np.cumsum(x)
    return (cumsum[2*n+1:] - cumsum[:-(n*2+1)]) / float(2*n+1)


def plot(xxs, yys, legends=None, x_label=None, y_label=None, save_path=None,
            smooth=True, scale=None):
    if not isinstance(xxs, list):
        xxs = [xxs]
    if not isinstance(yys, list):
        yys = [yys]
    if not isinstance(legends, list):
        legends = [legends]

    with plt.style.context("ggplot"):
        fig, ax = plt.subplots()
        for xx, yy, leg in zip(xxs, yys, legends):
            if smooth:
                yy = running_mean(yy, 5)
            ax.plot(xx, yy, label=leg)
            ax.set_xlabel(x_label)
            ax.set_ylabel(x_label)
        ax.legend()
        if scale is not None:
            ax.set_ylim(scale)
    if save_path is not None:
        return save_fig(fig, save_path)
    return fig
