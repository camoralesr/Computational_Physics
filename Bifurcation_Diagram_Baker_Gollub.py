import numpy as np
from numpy import sin, cos, pi
from fractions import Fraction
import matplotlib.pyplot as plt

# -----------------------------------------------------
# System parameters
# -----------------------------------------------------
q = 2
om = Fraction(2, 3)
T = 2*pi / om

# -----------------------------------------------------
# Initial conditions
# -----------------------------------------------------
x0 = 1.25
v0 = 0

# -----------------------------------------------------
# Gamma values
# -----------------------------------------------------
gamma_min = 0.9
gamma_max = 1.8
dgamma = 0.0001
gamma_values = np.arange(gamma_min, gamma_max + dgamma, dgamma)
n_orbits = len(gamma_values)

# -----------------------------------------------------
# Numerical method parameters
# -----------------------------------------------------
Trans = 300
Nkeep = 600
steps_per_T = 300
dt = T / steps_per_T

# -----------------------------------------------------
# Initial state
# -----------------------------------------------------
x = np.full(n_orbits, x0)
v = np.full(n_orbits, v0)
y = np.concatenate([x, v])

# -----------------------------------------------------
# Dynamics
# -----------------------------------------------------
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:]
    dx = v
    dv = -(1/q)*v - sin(x) + gamma_values*cos(om*t)
    return np.concatenate([dx, dv])

# -----------------------------------------------------
# Fourth-order Runge-Kutta method
# -----------------------------------------------------
def rk4(f, t, y, h):
    k1 = h*f(t, y)
    k2 = h*f(t + h/2, y + k1/2)
    k3 = h*f(t + h/2, y + k2/2)
    k4 = h*f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6

# -----------------------------------------------------
# Storage for stroboscopic points
# -----------------------------------------------------
x_strobe = np.empty((Nkeep, n_orbits))
v_strobe = np.empty((Nkeep, n_orbits))
save_index = 0

# -----------------------------------------------------
# Integration
# -----------------------------------------------------
total_periods = Trans + Nkeep
total_steps = total_periods * steps_per_T
for step in range(total_steps):
    current_time = step*dt
    y = rk4(dyn, current_time, y, dt)
    completed_period = (step + 1) // steps_per_T
    if (step + 1) % steps_per_T == 0:
        if completed_period > Trans:
            x_strobe[save_index] = y[:n_orbits]
            v_strobe[save_index] = y[n_orbits:]
            save_index += 1

# -----------------------------------------------------
# Bifurcation diagram & Figure format
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(n_orbits):
    ax.scatter(np.full(Nkeep, gamma_values[i]), 
            v_strobe[:, i], s=0.5, color='blue', linewidths=0, rasterized=True)
ax.set_xlabel(r'$\gamma$', fontsize=16)
ax.set_ylabel(r'$\dot{\theta}$', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.text(0.03, 0.97, rf'$q = {q}$',
        transform=ax.transAxes, ha='left', va='top', fontsize=14)
ax.text(0.03, 0.03, rf'$\omega_D = {om}$',
        transform=ax.transAxes, ha='left', va='bottom', fontsize=14)
ax.set_xlim(gamma_min, gamma_max)
ax.set_box_aspect(0.65)
plt.tight_layout()

# -----------------------------------------------------
# Save figure
# -----------------------------------------------------
plt.savefig('Bif.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()