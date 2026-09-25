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
x0_values = np.arange(-4, 4.1, 0.1)
v0_values = np.arange(0, 0.5, 0.5)
X0, V0 = np.meshgrid(x0_values, v0_values)
x0_flat = X0.ravel()
v0_flat = V0.ravel()
n_orbits = len(x0_flat)
y = np.concatenate([x0_flat, v0_flat])

# -----------------------------------------------------
# Method parameters
# -----------------------------------------------------
Trans = 0
Nperiods = 5000
steps_per_T = 300
dt = T / steps_per_T

# -----------------------------------------------------
# Dynamics: forced Duffing oscillator
# -----------------------------------------------------
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
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
n_saved = Nperiods - Trans + 1
x_strobe = np.empty((n_saved, n_orbits))
v_strobe = np.empty((n_saved, n_orbits))
# Initial stroboscopic point
save_index = 0
if Trans == 0:
    x_strobe[save_index] = y[:n_orbits]
    v_strobe[save_index] = y[n_orbits:]
    save_index += 1

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
            x_strobe[save_index] = y[:n_orbits]
            v_strobe[save_index] = y[n_orbits:]
            save_index += 1

# ---------------------------------------------------------
# Colors according to initial condition & Stroboscopic map
# ---------------------------------------------------------
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
orbit_colors = [colors[i % len(colors)] for i in range(n_orbits)]
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(n_orbits):
    ax.scatter(x_strobe[:, i], v_strobe[:, i], s=1, color=orbit_colors[i],
        linewidths=0, rasterized=True)
ax.set_xlabel(r"$x$", fontsize=22)
ax.set_ylabel(r"$\dot{x}$", fontsize=22)
ax.tick_params(axis="both", labelsize=18)
ax.set_box_aspect(0.65)
plt.tight_layout()
plt.savefig('MS.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()