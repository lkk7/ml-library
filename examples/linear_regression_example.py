import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from linear_models.linear_regression import LinearRegression
plt.style.use('seaborn-whitegrid')

# Simple data.
n_rows = 50
x = np.linspace(0, 1, n_rows)
y = 0.5 - 1.2 * x + 1.4 * x**2 + np.random.normal(0, 0.05, n_rows)
x_poly = np.column_stack((x, x**2))

# Gradient descent method
model_grad = LinearRegression(learning_rate=0.5, method='gradient', warm_start=True)
model_grad.fit(x_poly, y, n_iter=1)
# Stochastic gradient descent method
model_sgd = LinearRegression(learning_rate=0.1, method='sgd', warm_start=True)
model_sgd.fit(x_poly, y, n_iter=1)

fig, ax = plt.subplots(1, 2, figsize=(7, 3))
line1, = ax[0].plot(x, model_grad.predict(x_poly), c='r', lw=3)
line2, = ax[1].plot(x, model_sgd.predict(x_poly), c='r', lw=3)
for i in range(2):
    ax[i].scatter(x, y, c='navy')
    ax[i].set_xlabel('$x$')
    ax[i].set_ylabel('$y$')
    ax[i].set_ylim(0, 1)
ax[0].set_title('Gradient descent')
ax[1].set_title('Stochastic gradient descent')
plt.tight_layout()


grad_solutions = []
sgd_solutions = []
for i in range(0, 5000, 250):
    sgd_iter = int(i / 4)
    model_grad.fit(x_poly, y, n_iter=100)
    model_sgd.fit(x_poly, y, n_iter=sgd_iter)
    grad_solutions.append(model_grad.predict(x_poly))
    sgd_solutions.append(model_sgd.predict(x_poly))


def update(t):
    line1.set_ydata(grad_solutions[t])
    line2.set_ydata(sgd_solutions[t])


anim = FuncAnimation(fig, update, frames=range(len(grad_solutions)), interval=100)
plt.show()
