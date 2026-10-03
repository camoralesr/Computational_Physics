import numpy as np
from numpy import sin, cos, pi, log
import matplotlib.pyplot as plt

# -----------------------------------------------------
# System parameters
# -----------------------------------------------------
q = 2.0
om = 2/3
T = 2*pi / om

# -----------------------------------------------------
# Initial conditions
# -----------------------------------------------------
x0 = 1.25
v0 = 0
t = 0

# -----------------------------------------------------
# Gamma values
# -----------------------------------------------------
gamma_values = np.arange(0.9, 1.6, 0.0001)
n_orbits = len(gamma_values)

# -----------------------------------------------------
# Method parameters
# -----------------------------------------------------
nTrans = 300
nLyap = 300
steps_per_T = 300
dt = T / steps_per_T

# -----------------------------------------------------
# Initial state & unit tangent vector
# -----------------------------------------------------
x = np.full(n_orbits, x0)
v = np.full(n_orbits, v0)
xi = np.full(n_orbits, 1.0/np.sqrt(2.0))
eta = np.full(n_orbits, 1.0/np.sqrt(2.0))
y = np.concatenate([x, v, xi, eta])

# -----------------------------------------------------
# Dynamics + variational equations
# -----------------------------------------------------
def dyn(t, y):
    x = y[:n_orbits]
    v = y[n_orbits:2*n_orbits]
    xi = y[2*n_orbits:3*n_orbits]
    eta = y[3*n_orbits:]
    dx = v
    dv = (-(1/q)*v - sin(x) + gamma_values*cos(om*t))
    dxi = eta
    deta = (-cos(x)*xi - (1/q)*eta)
    return np.concatenate([dx, dv, dxi, deta])

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
# Integration: transient regime
# -----------------------------------------------------
for period in range(nTrans):
    for step in range(steps_per_T):
        y = rk4(dyn, t, y, dt)
        t += dt
    xi = y[2*n_orbits:3*n_orbits]
    eta = y[3*n_orbits:]
    tangent_norm = np.sqrt(xi**2 + eta**2)
    y[2*n_orbits:3*n_orbits] = (xi / tangent_norm)
    y[3*n_orbits:] = (eta / tangent_norm)

# -----------------------------------------------------
# Maximum Lyapunov exponent
# -----------------------------------------------------
sum_log = np.zeros(n_orbits)
for period in range(nLyap):
    for step in range(steps_per_T):
        y = rk4(dyn, t, y, dt)
        t += dt
    xi = y[2*n_orbits:3*n_orbits]
    eta = y[3*n_orbits:]
    tangent_norm = np.sqrt(xi**2 + eta**2)
    sum_log += log(tangent_norm)
    y[2*n_orbits:3*n_orbits] = (xi / tangent_norm)
    y[3*n_orbits:] = (eta / tangent_norm)

# -----------------------------------------------------
# Maximum Lyapunov exponent per unit time
# -----------------------------------------------------
lambda_max = sum_log / (nLyap*T)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(gamma_values, lambda_max, linewidth=0.8, color='blue')
ax.axhline(y=0, linewidth=0.5, color='black')
ax.set_xlabel(r'$\gamma$', fontsize=22)
ax.set_ylabel(r'$\lambda_{\max}$', fontsize=22)
ax.tick_params(axis='both', labelsize=18)
ax.set_xlim(gamma_values[0], gamma_values[-1])
ax.set_box_aspect(0.65)
plt.tight_layout()


# -----------------------------------------------------
# Save figure
# -----------------------------------------------------
plt.savefig('Lya.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()