from cvxpy import *
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.affine.reshape import reshape
from imageio import imread
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(21)


def killed_pixels(shape, frac_kill):
    npixels = np.prod(shape)
    num_kill = int(frac_kill * npixels)
    inds = np.random.choice(npixels, num_kill, replace=False)
    return inds


def get_noisy_data():
    fname = '../images/monalisa.png'
    original = np.mean(imread(fname), axis=-1)
    shape = original.shape

    total = range(np.prod(shape))
    unknown = killed_pixels(shape, frac_kill=0.8)
    known = list(set(total) - set(unknown))

    corrupted = np.zeros(shape)
    rows, cols = np.unravel_index(known, shape)
    corrupted[rows, cols] = original[rows, cols]
    return original, corrupted


def get_text_data():
    fname_1 = '../images/monalisa.png'
    fname_2 = '../images/text.png'

    original = np.mean(imread(fname_1), axis=-1)
    text = np.mean(imread(fname_2), axis=-1)
    corrupted = np.minimum(original + text, 255)
    return original, corrupted


def get_regular_noisy_data():
    fname = '../images/monalisa.png'
    original = np.mean(imread(fname), axis=-1)
    corrupted = original.copy()
    for i in [3, 4, 5, 7, 11]:
        corrupted[0::i, 0::i] = 0
    return original, corrupted


def total_variation(x):
    dx = x[1:, :-1] - x[:-1, :-1]
    dy = x[:-1, 1:] - x[:-1, :-1]
    D = vstack((vec(dx), vec(dy)))
    cost = sum(norm(D, p=1, axis=0))
    return cost


def inpaint(corrupted, rows, cols):
    x = Variable(corrupted.shape)
    objective = Minimize(total_variation(x))

    knowledge = x[rows, cols] == corrupted[rows, cols]
    constraints = [0 <= x, x <= 255, knowledge]

    prob = Problem(objective, constraints)
    prob.solve(verbose=True, solver=SCS)
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


def main():
    modes = ['text', 'noisy', 'regular']
    data_funs = [get_text_data, get_noisy_data, get_regular_noisy_data]

    for data_fun, mode in zip(data_funs, modes):
        original, corrupted = data_fun()
        rows, cols = np.where(original == corrupted)
        recovered = inpaint(corrupted, rows, cols)
        fname = f'../images/mona_lisa_{mode}_results.png'
        compare(corrupted, recovered, original, fname)


if __name__ == '__main__':
    main()
