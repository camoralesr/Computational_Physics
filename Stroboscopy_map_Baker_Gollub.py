import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
from fractions import Fraction

# -----------------------------------------------------
# System parameters
# -----------------------------------------------------
q = 2
om = Fraction(2, 3)
gamma = 1.1799
T = 2*pi / om

# -----------------------------------------------------
# Initial conditions: grid
# -----------------------------------------------------
x0_values = np.arange(-pi, pi, 0.1)
v0_values = np.arange(-2, 2.2, 0.1)
X0, V0 = np.meshgrid(x0_values, v0_values)
x0_flat = X0.ravel()
v0_flat = V0.ravel()
n_orbits = len(x0_flat)
y = np.concatenate([x0_flat, v0_flat])

# -----------------------------------------------------
# Method parameters
# -----------------------------------------------------
Trans = 10
Nperiods = 2000
steps_per_T = 300
dt = T / steps_per_T

# -----------------------------------------------------
# Dynamics
# -----------------------------------------------------
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dx = v
    dv = -(1/q)*v - sin(x) + gamma*cos(om*t)
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
            theta = y[:n_orbits]
            theta_wrapped = (theta + pi) % (2*pi) - pi
            x_strobe[save_index] = theta_wrapped
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
ax.set_xlabel(r"$\theta$", fontsize=16)
ax.set_ylabel(r"$\dot{\theta}$", fontsize=16)
ax.tick_params(axis="both", labelsize=12)
ax.text(0.03, 0.97, rf'$q = {q}$',
        transform=ax.transAxes, ha='left', va='top', fontsize=14)
ax.text(0.03, 0.03, rf'$\omega_D = {om}$',
        transform=ax.transAxes, ha='left', va='bottom', fontsize=14)
ax.text(0.97, 0.97, rf'$\gamma = {gamma}$',
        transform=ax.transAxes, ha='right', va='top', fontsize=14)
ax.set_box_aspect(0.65)
plt.tight_layout()
plt.savefig('MS.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()



# ---------------------------------------------------------
# Algunos resultados
# ---------------------------------------------------------
print(f"Periodo = {T:.6f}")
print(f"paso del método = {dt:.6f}")
print(f"paso de integración = {current_time:.6f}")