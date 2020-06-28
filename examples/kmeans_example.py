import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from cluster.kmeans import KMeans
plt.style.use('seaborn-whitegrid')

# Simple data generation
n_rows = 1000
x = np.random.normal(0, 0.15, (n_rows, 2))
for i, v in zip(range(1, 5), np.array([[-1, 1], [1, 1], [0, -1.5], [0, -3.5]])):
    x[int((i-1) * n_rows / 4):int(i * n_rows / 4), :] += v * 0.3

colors = np.array(('skyblue', 'gold', 'lime', 'crimson'))
ks = [2, 3, 4]
models_colors = []
fig, ax = plt.subplots(1, 3, figsize=(7, 3))
for i, k_i in enumerate(ks):
    models_colors.append(colors[KMeans(k=k_i).fit_predict(x, n_iter=1)])
    ax[i].scatter(*x.T, c=models_colors[i])
    ax[i].set_title('k = {}'.format(k_i), size=15)
    ax[i].tick_params(labelbottom=False, labelleft=False)

plt.tight_layout()
plt.show()