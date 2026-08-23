import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------------------------------
# System parameters
# -----------------------------------------------------
g = 9.81 
l1 = 2
l2 = 1 
m1 = 4 
m2 = 1

# -----------------------------------------------------
# Initial conditions
# -----------------------------------------------------
th10 = np.radians(90.0)
th20 = np.radians(180.0)
ome10, ome20 = 0, 0

# -----------------------------------------------------
# Method parameters 
# -----------------------------------------------------
tmax = 20          # tiempo simulado 
dt = 0.01          # paso de RK4      
TRAIL = 500        # puntos de estela de m2
STRIDE = 12        # se muestra 1 de cada STRIDE pasos

# -----------------------------------------------------
# Dynamics: Double pendulum
# -----------------------------------------------------
def dyn(t, y):
    th1, w1, th2, w2 = y
    d = th1 - th2
    sd = sin(d)
    cd = cos(d)
    den = 2*m1 + m2 - m2*cos(2*th1 - 2*th2)
    a1 = (-g*(2*m1 + m2)*sin(th1) - g*m2*sin(th1 - 2*th2) - 2*m2*sd*(l1*cd*w1**2 + l2*w2**2))/(l1*den)
    a2 = (2*sd*(g*(m1 + m2)*cos(th1) + l1*(m1 + m2)*w1**2 + l2*m2*cd*w2**2))/(l2*den)
    return np.array([w1, a1, w2, a2])

# -----------------------------------------------------
# Fourth-order Runge-Kutta method
# -----------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# -----------------------------------------------------
# Integration using RK4
# -----------------------------------------------------
n = int(tmax / dt)
t = np.linspace(0, n * dt, n + 1)
y = np.empty((n + 1, 4))
y[0] = np.array([th10, ome10, th20, ome20])
for i in range(n):
    y[i + 1] = rk4(dyn, t[i], y[i], dt)

# -----------------------------------------------------
# Separate variables after integration
# -----------------------------------------------------
th1  = y[:, 0]
ome1 = y[:, 1]
th2  = y[:, 2]
ome2 = y[:, 3]

# -----------------------------------------------------
# Kinematics
# -----------------------------------------------------
x1, y1 = l1 * sin(th1), -l1 * cos(th1)
x2, y2 = x1 + l2 * sin(th2), y1 - l2 * cos(th2)

# -----------------------------------------------------
# Figure
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
R = l1 + l2 + 0.2
ax.set(xlim=(-R, R),ylim=(-R, R),aspect="equal",title="Double pendulum (RK4)")
ax.title.set_fontsize(18)
ax.tick_params(axis="both", labelsize=12)
ax.grid(alpha=0.3)
line, = ax.plot([], [], "o-", lw=1.3, color="blue")
trace, = ax.plot([], [], "-", lw=1, alpha=0.6, color="red")
clock = ax.text(0.05, 0.93, "", transform=ax.transAxes, fontsize=12)

# -----------------------------------------------------
# Animation function
# -----------------------------------------------------
def animate(i):
    line.set_data([0, x1[i], x2[i]],[0, y1[i], y2[i]])
    trace.set_data(x2[:i+1], y2[:i+1])
    clock.set_text(f"t = {t[i]:.1f} s")
    return line, trace, clock

# -----------------------------------------------------
# Create animation
# -----------------------------------------------------
ani = FuncAnimation(fig, animate, frames=range(0, n + 1, STRIDE), 
                interval=STRIDE * dt * 1000, blit=True)
plt.tight_layout()
plt.show()


#def rkf45_step(f, t, y, h):
#    k1 = h * f(t, y)
#    k2 = h * f(t + h/4, y + k1/4)
#    k3 = h * f(t + 3*h/8, y + 3*k1/32 + 9*k2/32)
#    k4 = h * f(t + 12*h/13, y + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
#    k5 = h * f(t + h, y + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
#    k6 = h * f(t + h/2, y - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)
#    y4 = y + (25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5)
#    return y + (16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55)