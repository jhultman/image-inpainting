from functools import partial
from multiprocessing import Pool
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from inpainting import inpaint
np.random.seed(21)


def partial_ravel(arr, lo=0, hi=-1):
    """Ravel axes in [lo, hi)."""
    assert (lo < hi) or (hi == -1)
    shape = arr.shape[:lo] + (-1,) + arr.shape[hi:-1]
    return arr.reshape(shape)


def paste_region(arr, region, val):
    """Make copy of arr with arr[region] == val."""
    arr = arr.copy()
    arr[region] = val
    return arr


def region_to_mask(shape, region):
    """Make arr with arr[region] == True."""
    mask = np.full(shape, False)
    mask[region] = True
    return mask


def _imshow(ax, img):
    ax.imshow(img)
    ax.set_axis_off()


def plot_compare(corrupted, recovered, region):
    difference = (corrupted - recovered)[region]
    fig = plt.figure(figsize=(11, 6))
    ax = partial(plt.subplot2grid, shape=(7, 4), colspan=2)
    _imshow(ax(loc=(0, 0), rowspan=6), corrupted)
    _imshow(ax(loc=(0, 2), rowspan=6), recovered)
    _imshow(ax(loc=(6, 1), rowspan=1), difference)
    fig.tight_layout()
    fig.savefig('../images/readme/watermark_results.png')


def get_stacked_images():
    """Images stacked in last dim."""
    basedir = '../images/watermarked'
    fnames = [f'{basedir}/stock{i}.jpg' for i in range(3)]
    img = np.stack([imread(fname) for fname in fnames], axis=-1)
    return img


def get_watermark_mask(all_img, lo=60, hi=256):
    """Thresholding by RGB pixel vals."""
    mask = (all_img > lo) & (all_img < hi)
    mask = partial_ravel(mask, 2).all(-1)
    return mask


def get_watermark_region():
    """Crudely localized by hand."""
    r1, r2, c1, c2 = 340, 380, 135, 475
    region = (slice(r1, r2), slice(c1, c2))
    return region


def solve_problem(img, rows, cols):
    """Multiprocess wrapper for RGB inpainting."""
    task = partial(inpaint, rows=rows, cols=cols, verbose=True)
    channels_first = tuple(np.rollaxis(img, -1))
    with Pool(3) as pool:
        recovered = pool.map(task, channels_first)
    recovered = np.stack(recovered, -1).astype(np.uint8)
    return recovered


def main():
    all_img = get_stacked_images()
    sample_img = all_img[..., 0]
    mask = get_watermark_mask(all_img)
    region = get_watermark_region()
    mask &= region_to_mask(mask.shape, region)
    cropped_img = sample_img[region]
    rows, cols = np.where(~mask[region])
    recovered = solve_problem(sample_img, rows, cols)
    recovered_img = paste_region(sample_img, region, recovered)
    plot_compare(sample_img, recovered_img, region)


if __name__ == '__main__':
    main()
