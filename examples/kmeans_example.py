import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from cluster.kmeans import KMeans
plt.style.use('seaborn-whitegrid')


# Simple data generation
n_rows = 500
x = np.random.normal(0, 0.15, (n_rows, 2))
for i, v in zip(range(1, 5), np.array([[-1, 1], [1, 1], [0, -1.5], [0, -3.5]])):
    x[int((i-1) * n_rows / 4):int(i * n_rows / 4), :] += v * 0.3

colors = np.array(('skyblue', 'gold', 'lime', 'crimson'))
ks = [2, 3, 4]
models = []
models_colors = []
plots = []
fig, ax = plt.subplots(1, 3, figsize=(7, 3))
for i, k_i in enumerate(ks):
    models.append(KMeans(k=k_i, warm_start=True))
    plots.append(ax[i].scatter(*x.T, c=colors[models[i].fit_predict(x, n_iter=1)], s=15))
    ax[i].set_title('k = {}'.format(k_i), size=15)
    ax[i].tick_params(labelbottom=False, labelleft=False)
solutions = [[colors[models[i].fit_predict(x, n_iter=1)] for i in range(3)] for j in range(30)]


def update_plot(t):
    for j in range(3):
        plots[j].set_color(solutions[t][j])


plt.tight_layout()
anim = FuncAnimation(fig, update_plot, frames=len(solutions), interval=100)
plt.show()
