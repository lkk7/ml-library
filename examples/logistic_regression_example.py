import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from linear_models.logistic_regression import LogisticRegression
plt.style.use('seaborn-whitegrid')


def update_plot(t):
    plots[1].clear()
    plots[2].clear()
    plots[1].scatter(*x.T, c=grad_solutions[t])
    plots[2].scatter(*x.T, c=sgd_solutions[t])
    plots[1].set_title("Gradient descent")
    plots[2].set_title("Stochastic gradient descent")


# Simple data generation
n_rows = 200
x = np.random.normal(0, 0.05, (n_rows, 3))
y = np.zeros(n_rows)
sample = np.random.choice(x.shape[0], int(n_rows / 2), replace=False)
x[sample, :-1] += 0.15
y[sample] = 1

# Gradient descent method
model_grad = LogisticRegression(learning_rate=0.01, method='gradient', warm_start=True)
model_grad.fit(x, y, n_iter=1)
# Stochastic gradient descent method
model_sgd = LogisticRegression(learning_rate=0.01, method='sgd', warm_start=True)
model_sgd.fit(x, y, n_iter=1)

fig = plt.figure(figsize=(9, 3))
plots = [fig.add_subplot(131, projection=Axes3D.name),
         fig.add_subplot(132, projection=Axes3D.name),
         fig.add_subplot(133, projection=Axes3D.name)]
plots[0].scatter(*x.T, c=['crimson' if y_i == 1 else 'indigo' for y_i in y])
plots[1].scatter(*x.T, c=['crimson' if y_i == 1 else 'indigo' for y_i in model_grad.predict(x)])
plots[2].scatter(*x.T, c=['crimson' if y_i == 1 else 'indigo' for y_i in model_sgd.predict(x)])
plots[0].set_title("True labels")
for i in range(3):
    plots[i].tick_params(labelbottom=False, labelleft=False)
plt.tight_layout()

grad_solutions = []
sgd_solutions = []
for i in range(20):
    model_grad.fit(x, y, n_iter=(1 + i**2))
    model_sgd.fit(x, y, n_iter=500)
    grad_solutions.append(['crimson' if y_i == 1 else 'indigo' for y_i in model_grad.predict(x)])
    sgd_solutions.append(['crimson' if y_i == 1 else 'indigo' for y_i in model_sgd.predict(x)])

anim = FuncAnimation(fig, update_plot, frames=range(len(grad_solutions)), interval=100)
plt.show()
