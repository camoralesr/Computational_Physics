import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt

# -----------------------------------------------------
# System parameters
# -----------------------------------------------------
m = 1.0
k = 1.0
l = 1.0
F = 1.0
om = (2/3)*pi
T = 2*pi / om

# -----------------------------------------------------
# Initial conditions: grid
# -----------------------------------------------------
x0 = np.linspace(-3, 3, 100)
v0 = np.linspace(0, 1, 2)
X0, V0 = np.meshgrid(x0, v0)
x01 = X0.ravel()
v01 = V0.ravel()
n = len(x01)
print(n)
y = np.concatenate([x01, v01])

# -----------------------------------------------------
# Method parameters
# -----------------------------------------------------
Trans = 0
Nperiods = 1000
steps_per_T = 300
dt = T / steps_per_T

# -----------------------------------------------------
# Dynamics: forced Duffing oscillator
# -----------------------------------------------------
def dyn(t, y):
    x = y[:n]
    v = y[n:]
    dx = v
    dv = F*cos(om*t)/m - (k/m)*x - (l/m)*x**3
    return np.concatenate([dx, dv])

# -----------------------------------------------------
# Fourth-order Runge-Kutta method
# -----------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# -----------------------------------------------------
# Stroboscopic storage
# -----------------------------------------------------
ns = Nperiods - Trans + 1
xs = np.empty((ns, n))
vs = np.empty((ns, n))
s = 0
if Trans == 0:
    xs[s] = y[:n]
    vs[s] = y[n:]
    s += 1

# -----------------------------------------------------
# Integration
# -----------------------------------------------------
total_steps = Nperiods * steps_per_T
for step in range(total_steps):
    current_time = step*dt
    y = rk4(dyn, current_time, y, dt)
    completed_period = (step + 1) // steps_per_T
    if (step + 1) % steps_per_T == 0:
        if completed_period >= max(1, Trans):
            xs[s] = y[:n]
            vs[s] = y[n:]
            s += 1

# ---------------------------------------------------------
# Colors according to initial condition & Stroboscopic map
# ---------------------------------------------------------
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
oc = [colors[i % len(colors)] for i in range(n)]
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(n):
    ax.scatter(xs[:, i], vs[:, i], s=1, color=oc[i],
        linewidths=0, rasterized=True)
ax.set_xlim(-3, 3)
ax.set_ylim(-6, 6)
ax.set_xlabel(r"$x$", fontsize=14)
ax.set_ylabel(r"$\dot{x}$", fontsize=14)
ax.tick_params(axis="both", labelsize=12)
ax.set_box_aspect(0.6)
plt.tight_layout()
plt.savefig('MS.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()
