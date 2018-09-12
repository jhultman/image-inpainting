import numpy as np
import cvxpy as cp

from multiprocessing import Pool
from imageio import imread
import matplotlib.pyplot as plt

np.random.seed(21)


def killed_pixels(shape, frac_kill):
    npixels = np.prod(shape)
    num_kill = int(frac_kill * npixels)
    inds = np.random.choice(npixels, num_kill, replace=False)
    return inds


def load_text():
    fname = '../images/text.png'
    text = np.mean(imread(fname_2), axis=-1)
    return text


def load_mona_lisa():
    fname = '../images/monalisa.png'
    mona = np.mean(imread(fname), axis=-1)
    return mona


def get_noisy_data():
    original = load_mona_lisa()
    shape = original.shape

    total = range(np.prod(shape))
    unknown = killed_pixels(shape, frac_kill=0.8)
    known = list(set(total) - set(unknown))

    corrupted = np.zeros(shape)
    rows, cols = np.unravel_index(known, shape)
    corrupted[rows, cols] = original[rows, cols]
    return original, corrupted


def get_text_data():
    original = load_mona_lisa()
    text = load_text()
    corrupted = np.minimum(original + text, 255)
    return original, corrupted


def get_regular_noisy_data():
    original = load_mona_lisa()
    corrupted = original.copy()
    for i in [3, 4, 5, 7, 11]:
        corrupted[0::i, 0::i] = 0
    return original, corrupted


def total_variation(arr):
    dx = cp.vec(arr[1:, :-1] - arr[:-1, :-1])
    dy = cp.vec(arr[:-1, 1:] - arr[:-1, :-1])
    D = cp.vstack((dx, dy))
    norm = cp.norm(D, p=1, axis=0)
    return cp.sum(norm)


def inpaint(corrupted, rows, cols):
    x = cp.Variable(corrupted.shape)
    objective = cp.Minimize(total_variation(x))

    knowledge = x[rows, cols] == corrupted[rows, cols]
    constraints = [0 <= x, x <= 255, knowledge]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True, solver=cp.SCS)
    return x.value


def compare(corrupted, recovered, original, fname):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 15))
    images = [corrupted, recovered, original, recovered - original]
    titles = ['Corrupted', 'Recovered', 'Original', 'Difference']
    
    for i, (image, title) in enumerate(zip(images, titles)):
        ax[i].imshow(image)
        ax[i].set_title(title)
        ax[i].set_axis_off()

    fig.tight_layout()
    plt.savefig(fname)


def task(data_fun, mode):
    original, corrupted = data_fun()
    rows, cols = np.where(original == corrupted)
    recovered = inpaint(corrupted, rows, cols)
    fname = f'../images/mona_lisa_{mode}_results.png'
    compare(corrupted, recovered, original, fname)


def main():
    modes = ['text', 'noisy', 'regular']
    data_funs = [get_text_data, get_noisy_data, get_regular_noisy_data]
    args = zip(data_funs, modes)

    with Pool(len(modes)) as pool:
        pool.starmap(task, args)


if __name__ == '__main__':
    main()
