from tqdm import trange
import numpy as np, numpy.random as nr
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_line(ax, line, color, linestyle='-', label=None):
    get_y = lambda x: -(line[0]*x + line[2]) / line[1]
    points = np.asarray([[0, 1], [get_y(0), get_y(1)]])
    ax.axline(points[:, 0], points[:, 1], color=color, linestyle=linestyle, label=label)


def scatter(ax, pts):
    ax.plot(pts[0, 0], pts[0, 1], 'o', color='red', markersize=12)
    ax.plot(pts[1, 0], pts[1, 1], 'o', color='blue', markersize=10)
    ax.plot(pts[2, 0], pts[2, 1], 'o', color='c', markersize=6)
    ax.plot(pts[3, 0], pts[3, 1], 'o', color='orange', markersize=8)


def get_xor_data():
    # xor + bias
    x = np.array([
        [0, 0, 1], [0, 1, 1],
        [1, 0, 1], [1, 1, 1]
    ])
    y = np.array([[0, 1, 1, 0]]).T
    return x, y


def main():
    x_all, y_all = np.linspace(-2, 3, 51), np.linspace(-2, 3, 51)
    grid = np.transpose([np.tile(x_all, len(y_all)), np.repeat(y_all, len(x_all))])
    grid_bias = np.ones((len(grid), 1))
    grid = np.concatenate((grid, grid_bias), axis=1)
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

    f, axes = plt.subplots(2, 3)
    axes = axes.flatten()
    idx_correct = None
    for i in trange(n_epochs):
        x0 = x @ w0
        s0 = sigmoid(x0)
        x1 = s0 @ w1
        s1 = sigmoid(x1)

        grid_0 = sigmoid(grid @ w0)
        grid_1 = sigmoid(grid_0 @ w1)

        if idx_correct is None and (np.round(s1) == y).all():
            idx_correct = i

        # loss
        diff = y - s1  # dL/dO
        losses[i] = (diff ** 2).mean()

        # dσ/dx
        ds1 = s1 * (1 - s1)
        ds0 = s0 * (1 - s0)

        # dL/dx
        dx1 = ds1 * diff
        dx0 = ds0 * (dx1 @ w1.T)

        # dL/dw
        grad0 = x.T  @ dx0
        grad1 = s0.T @ dx1
        grad0_history.iloc[i] = grad0.T.flatten()
        grad1_history.iloc[i] = grad1.flatten()
        
        # step
        w0 += lr * grad0
        w1 += lr * grad1

        def update_vis():
            for ax in axes:
                ax.clear()
            ax1, ax2, ax3, ax4, ax5, ax6 = axes

            ###############################################################
            ax1.set_title('1. First neurons')

            b_grid = grid[(grid_1 >  0.5).flatten(), :2]
            r_grid = grid[(grid_1 <= 0.5).flatten(), :2]
            ax1.scatter(b_grid[:, 0], b_grid[:, 1], s=1, c='b')
            ax1.scatter(r_grid[:, 0], r_grid[:, 1], s=1, c='r')

            scatter(ax1, x)
            ax1.set(xlim=(-2, 3), ylim=(-2, 3))

            line0, line1 = w0[:, 0], w0[:, 1]
            norm0 = np.linalg.norm(line0[:-1])
            norm1 = np.linalg.norm(line1[:-1])
            
            # σ(-4.6) ~= 0.01; σ(4.6) ~= 0.99
            plot_line(ax1, line0,               color='green', label=f'norm={norm0:0.3f}')
            plot_line(ax1, line0 + [0, 0, 4.6], color='green', linestyle=':')
            plot_line(ax1, line0 - [0, 0, 4.6], color='green', linestyle=':')

            plot_line(ax1, line1,               color='yellow', label=f'norm={norm1:0.3f}')
            plot_line(ax1, line1 + [0, 0, 4.6], color='yellow', linestyle=':')
            plot_line(ax1, line1 - [0, 0, 4.6], color='yellow', linestyle=':')
            ax1.legend()

            ###############################################################
            ax2.set_title('2. Linear transform')
            
            scatter(ax2, x0)
            x0_min, x0_max = x0.min(axis=0), x0.max(axis=0)
            ax2.set(
                xlim=(min(x0_min[0], 0) - 0.1, x0_max[0] + 0.1),
                ylim=(min(x0_min[1], 0) - 0.1, x0_max[1] + 0.1)
            )

            ax2.axvline(0,    linestyle='--', color='green')
            ax2.axvline(4.6,  linestyle=':',  color='green')
            ax2.axvline(-4.6, linestyle=':',  color='green')

            ax2.axhline(0,    linestyle='--', color='yellow')
            ax2.axhline(4.6,  linestyle=':',  color='yellow')
            ax2.axhline(-4.6, linestyle=':',  color='yellow')

            ###############################################################
            ax3.set_title('3. After sigmoid. 2nd layer')
            
            scatter(ax3, s0)
            s0_min, s0_max = s0.min(axis=0), s0.max(axis=0)
            ax3.set(
                xlim=(s0_min[0] - 0.1, s0_max[0] + 0.1),
                ylim=(s0_min[1] - 0.1, s0_max[1] + 0.1)
            )
            
            ax3.axvline(0.5,  linestyle='--', color='green')
            ax3.axvline(0.01, linestyle=':',  color='green')
            ax3.axvline(0.99, linestyle=':',  color='green')

            ax3.axhline(0.5,  linestyle='--', color='yellow')
            ax3.axhline(0.01, linestyle=':',  color='yellow')
            ax3.axhline(0.99, linestyle=':',  color='yellow')

            line2 = np.asarray(w1.flatten().tolist() + [0])
            plot_line(ax3, line2, color='red')
            plot_line(ax3, line2 + [0, 0, 4.6], color='red', linestyle=':')
            plot_line(ax3, line2 - [0, 0, 4.6], color='red', linestyle=':')

            ###############################################################
            ax4.set_title('4. After 2nd sigmoid. Final classifier')

            x4 = np.linspace(-3, 3, 121)
            y4 = sigmoid(x4)
            ax4.plot(x4, y4, ':k')
            ax4.axvline(0,   linestyle='--', color='red')
            ax4.axhline(0.5, linestyle='--', color='red')
            
            sizes = [12, 10, 6, 8]
            colors = ['red', 'blue', 'c', 'orange']
            for j in range(4):
                ax4.plot(x1[j], s1[j], 'o', color=colors[j], markersize=sizes[j])

            ax4.set(
                xlim=(x1.min() - 0.05, x1.max() + 0.05),
                ylim=(s1.min() - 0.05, s1.max() + 0.05)
            )

            ###############################################################
            ax5.set_title('5. Loss')
            
            ax5.plot(losses, 'k-', label=f'iter#{i}: {losses[i]:0.3f}')
            ax5.plot(losses[::50], 'ko')
            ax5.legend()

            ###############################################################
            ax6.set_title('6. Gradients')
            
            ax6.plot(grad0_history.dropna().iloc[:, :3], 'g-')
            ax6.plot(grad0_history.dropna().iloc[:, 3:], '-', color='yellow')
            ax6.plot(grad1_history.dropna(), 'r-')
            ax6.set(ylim=(-0.03, 0.02))
            
            ###############################################################
            if idx_correct is not None:
                ax5.axvline(idx_correct, linestyle='--')
                ax6.axvline(idx_correct, linestyle='--')

            f.show()
            if plt.waitforbuttonpress(1e-3) is not None:
                plt.waitforbuttonpress()

        update_vis()


if __name__ == '__main__':
    main()
    plt.waitforbuttonpress()
