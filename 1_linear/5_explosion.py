from tqdm import trange
import numpy as np, numpy.random as nr
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_xor_data():
    # xor + bias
    x = np.array([
        [0, 0, 1], [0, 1, 1],
        [1, 0, 1], [1, 1, 1]
    ])
    y = np.array([[0, 1, 1, 0]]).T
    return x, y


def main():
    x, y = get_xor_data()
    n_epochs = 600
    lr = 2

    nr.seed(0)
    w0 = 2 * nr.random((3, 2)) - 1
    w1 = 2 * nr.random((2, 1)) - 1

    index = range(n_epochs)
    losses = pd.Series(index=index, dtype=float)
    grad0_history = pd.DataFrame(index=index, columns=range(w0.size), dtype=float)
    grad1_history = pd.DataFrame(index=index, columns=range(w1.size), dtype=float)
    ds0_history   = pd.DataFrame(index=index, columns=range(8), dtype=float)
    ds1_history   = pd.DataFrame(index=index, columns=range(4), dtype=float)
    dx0_history   = pd.DataFrame(index=index, columns=range(8), dtype=float)
    dx1_history   = pd.DataFrame(index=index, columns=range(4), dtype=float)
    diff_history  = pd.DataFrame(index=index, columns=range(4), dtype=float)

    for i in trange(n_epochs):
        s0 = sigmoid(x @ w0)
        s1 = sigmoid(s0 @ w1)

        diff = y - s1
        losses[i] = (diff ** 2).mean()
        diff_history.iloc[i] = diff.flatten()

        ds1 = s1 * (1 - s1)
        ds0 = s0 * (1 - s0)
        ds1_history.iloc[i] = ds1.flatten()
        ds0_history.iloc[i] = ds0.flatten()

        dx1 = ds1 * diff
        dx0 = ds0 * (dx1 @ w1.T)
        dx0_history.iloc[i] = dx0.flatten()
        dx1_history.iloc[i] = dx1.flatten()

        grad0 = x.T @ dx0
        grad1 = s0.T @ dx1
        grad0_history.iloc[i] = grad0.T.flatten()
        grad1_history.iloc[i] = grad1.flatten()

        w0 += lr * grad0
        w1 += lr * grad1

    limits = range(420, 450), range(450, 480)

    histories = dict(
        loss=losses, diff=diff_history, grad0=grad0_history, grad1=grad1_history,
        ds0=ds0_history, ds1=ds1_history, dx0=dx0_history, dx1=dx1_history
    )

    f, axes = plt.subplots(2, 8)
    axes = axes.flatten()
    for i, (name, history) in enumerate(histories.items()):
        history.iloc[limits[0]].plot(ax=axes[2*i])
        history.iloc[limits[1]].plot(ax=axes[2*i+1])
        axes[2*i].set_title(f'{name} 420..450')
        axes[2*i+1].set_title(f'{name} 450..480')

    # losses.plot()
    # losses.cummin().plot()  # correct behaviour
    # loss_explosion = losses[losses > losses.cummin()]
    # loss_explosion.plot()  # explosion!
    # lims = loss_explosion.index.min(), loss_explosion.index.max()  # 458: 477


if __name__ == '__main__':
    main()
    plt.waitforbuttonpress()
