import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt

# -----------------------------------------------------
# System parameters
# -----------------------------------------------------
mu = 2.8
rmax = 1/(mu - 1)
drmax = np.sqrt(2/(1 + mu))

# -----------------------------------------------------
# Method parameters
# -----------------------------------------------------
Tmax = 1000
DT = 0.001
Trans = 10
r_stop = 0.001
omega_min = 1e-6
Nsteps = int(np.ceil(Tmax/DT))
dt = Tmax/Nsteps

# -----------------------------------------------------
# Initial conditions: grid
# -----------------------------------------------------
r0 = np.linspace(0, rmax, 25)
dr0 = np.linspace(-drmax, drmax, 25)
R0, DR0 = np.meshgrid(r0, dr0)

# -----------------------------------------------------
# Admissible initial conditions
# -----------------------------------------------------
r01 = R0.ravel()
dr01 = DR0.ravel()
rad = 2 - (1 + mu)*dr01**2 - 2*r01*(mu - 1)
valid = (r01 > r_stop) & (rad > 0)
r01 = r01[valid]
dr01 = dr01[valid]
w0_flat = np.sqrt(rad[valid])/r01
n = len(r01)
if n == 0:
    raise ValueError("No admissible initial conditions.")
y = np.concatenate([r01, np.zeros(n), dr01, w0_flat])

# -----------------------------------------------------
# Dynamics
# -----------------------------------------------------
def dyn(t, y):
    r = y[:n]
    th = y[n:2*n]
    dr = y[2*n:3*n]
    w = y[3*n:]
    safe_r = np.maximum(r, r_stop)
    d_r = dr
    d_th = w
    d_dr = (r*w**2 - mu + cos(th))/(1 + mu)
    d_w = -(2*dr*w + sin(th))/safe_r
    return np.concatenate([d_r, d_th, d_dr, d_w])

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
# Mechanical energy
# -----------------------------------------------------
def energy(r, th, dr, w):
    return (0.5*(1 + mu)*dr**2 + 0.5*r**2*w**2 + r*(mu - cos(th)))

# -----------------------------------------------------
# Poincare storage
# -----------------------------------------------------
r_points = [[] for _ in range(n)]
dr_points = [[] for _ in range(n)]
crossing_count = np.zeros(n, dtype=int)
active = np.ones(n, dtype=bool)
max_energy_error = np.zeros(n)

# -----------------------------------------------------
# Henon's correction at section crossings
# -----------------------------------------------------
def henon_crossing(r, th, dr, w, target):
    def g(theta, z):
        rr, vv, ww = z
        if rr <= r_stop or ww <= omega_min:
            raise ArithmeticError("Invalid crossing")
        return np.array([vv/ww, (rr*ww**2 - mu + cos(theta))/((1 + mu)*ww),
            -(2*vv*ww + sin(theta))/(rr*ww)])
    h = target - th
    z = np.array([r, dr, w], dtype=float)
    substeps = max(1, int(np.ceil(abs(h)/0.002)))
    hs = h/substeps
    for j in range(substeps):
        theta = th + j*hs
        z = rk4(g, theta, z, hs)
    if (z[0] <= r_stop or
        z[2] <= omega_min or
        not np.all(np.isfinite(z))):
        raise ArithmeticError("Invalid intersection")
    return z[0], z[1]

# -----------------------------------------------------
# Integration
# -----------------------------------------------------
for step in range(Nsteps):
    current_time = step*dt
    yold = y.copy()
    ynew = rk4(dyn, current_time, yold, dt)
    rold, thold = yold[:n], yold[n:2*n]
    drold, wold = yold[2*n:3*n], yold[3*n:]
    rnew, thnew = ynew[:n], ynew[n:2*n]    
    drnew, wnew = ynew[2*n:3*n], ynew[3*n:]    
    good = (np.isfinite(rnew) & np.isfinite(thnew) & np.isfinite(drnew)
         & np.isfinite(wnew) & (rnew > r_stop) & (rold > r_stop))
    active &= good
    target = 2*pi*np.ceil(thold/(2*pi)) # Detect upward crossings of theta = 2*pi*k
    crossed = (active & (thold < target - 1e-12) & (thnew >= target)
                & (wnew > omega_min) & (wold> omega_min))
    for i in np.flatnonzero(crossed):
        try: rr, vv = henon_crossing(rold[i], thold[i], drold[i], wold[i], target[i])
        except (ArithmeticError, FloatingPointError, OverflowError):
            continue
        crossing_count[i] += 1
        if crossing_count[i] > Trans:
            r_points[i].append(rr)
            dr_points[i].append(vv)   
    for part in range(4):  # Keep stopped trajectories inactive
        sl = slice(part*n, (part+1)*n)
        ynew[sl][~active], y = yold[sl][~active], ynew
    if step % 100 == 0:    # Monitor energy conservation
        err = abs(energy(rnew, thnew, drnew, wnew) - 1)
        max_energy_error[active] = np.maximum(
            max_energy_error[active],
            err[active])
    if not np.any(active):
        break

# ---------------------------------------------------------
# Colors according to initial condition & Poincare map
# ---------------------------------------------------------
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
orbit_colors = [colors[i % len(colors)] for i in range(n)]
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(n):
    if r_points[i]: 
        ax.scatter(r_points[i], dr_points[i], 
            s=1, color=orbit_colors[i], linewidths=0, rasterized=True)

# -----------------------------------------------------
# Boundary of the admissible region
# -----------------------------------------------------
rb = np.linspace(0, rmax, 400)
drb = np.linspace(-drmax, drmax, 400)
R, dR = np.meshgrid(rb, drb)
Z = (1 + mu)*dR**2 + 2*R*(mu - 1) - 2
ax.contour(R, dR, Z, levels=[0], colors='black',
        linewidths=0.2, linestyles='dashed')
ax.set_xlim(0, rmax)
ax.set_xlabel(r"$r$", fontsize=12)
ax.set_ylabel(r"$\dot{r}$", fontsize=12)
ax.tick_params(axis="both", labelsize=10)
ax.text(0.97, 0.97, rf'$\mu = {mu}$', transform=ax.transAxes,
    ha='right', va='top', fontsize=10)
ax.set_box_aspect(1)
plt.tight_layout()

plt.savefig('SAM.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()

# ---------------------------------------------------------
# Numerical results
# ---------------------------------------------------------
print(f"Integration step = {dt:.3f}")
print(f"Last integration time = {(step+1)*dt:.1f}")
print(f"Initial conditions = {n}")
print(f"Poincare points = {sum(map(len, r_points))}")
print(f"Stopped trajectories = {np.count_nonzero(~active)}")
print(f"Maximum sampled energy error = {np.max(max_energy_error):.3e}")
