import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------------------------------
# Parámetros del sistema
# -----------------------------------------------------
g = 9.81
m = 1.0
l = 2.0
R = 1.0
omega = 2

# Condiciones iniciales
phi0 = np.radians(180)
dphi0 = 0

# Integración temporal
tmax = 30
dt = 1e-3
STRIDE = 35
SPEED = 3

# -----------------------------------------------------
# Tamaños de letra
# -----------------------------------------------------
TITLE_SIZE = 20
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 14
CLOCK_SIZE = 14

# -----------------------------------------------------
# Dinámica: péndulo unido a un borde giratorio
# -----------------------------------------------------
def derivs(t, y):
    phi, w = y
    a = (R * omega**2 * cos(phi - omega*t) - g * sin(phi))/l
    return np.array([w, a])

# -----------------------------------------------------
# Método de Runge-Kutta de cuarto orden
# -----------------------------------------------------
def rk4_step(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# -----------------------------------------------------
# Energía
# -----------------------------------------------------
def energies(y, t):
    phi, dphi = y.T
    vbx = -R * omega * sin(omega*t) + l * cos(phi) * dphi
    vby =  R * omega * cos(omega*t) + l * sin(phi) * dphi
    K = 0.5 * m * (vbx**2 + vby**2)
    U = m * g * (R * sin(omega*t) - l * cos(phi))
    return K, U, K + U

# -----------------------------------------------------
# Integración con RK4
# -----------------------------------------------------
n = int(tmax / dt)
t = np.linspace(0, n * dt, n + 1)
y = np.empty((n + 1, 2))
# Condición inicial [phi, dphi]
y[0] = [phi0, dphi0]
for i in range(n):
    y[i + 1] = rk4_step(derivs, t[i], y[i], dt)
K, U, E = energies(y, t)

# -----------------------------------------------------
# Separar las variables después de integrar
# -----------------------------------------------------
phi = y[:, 0]
dphi = y[:, 1]

# -----------------------------------------------------
# Cinemática
# -----------------------------------------------------
xp = R * cos(omega * t)
yp = R * sin(omega * t)
xb = xp + l * sin(phi)
yb = yp - l * cos(phi)

# -----------------------------------------------------
# Figura
# -----------------------------------------------------
fig, (axp, axe) = plt.subplots(2, 1, figsize=(7, 10), 
            gridspec_kw={"height_ratios": [2.2, 1]})

# -----------------------------------------------------
# Panel del péndulo
# -----------------------------------------------------
radio_total = R + l + 0.2
axp.set_xlim(-radio_total, radio_total)
axp.set_ylim(-radio_total, radio_total)
axp.set_aspect("equal")
axp.set_title("Simple Pendulum Attached to a Rotating Rim",fontsize=TITLE_SIZE)
axp.grid(alpha=0.3)
axp.tick_params(axis="both",labelsize=TICK_SIZE)

# -----------------------------------------------------
# Disco guía
# -----------------------------------------------------
theta = np.linspace(0, 2*pi, 400)
axp.plot(R*cos(theta), R*sin(theta), lw=1.2, color="black")
# Radio del disco: línea discontinua
radius_line, = axp.plot([], [], "--", lw=1.2, color="black")
# Barra del péndulo
line, = axp.plot([], [], "-", lw=1.2, color="blue")

# -----------------------------------------------------
# Masa - Trayectoria - Reloj
# -----------------------------------------------------
bob, = axp.plot([], [], "o", ms=10, color="black")
trace, = axp.plot([], [], "-", lw=1, alpha=0.45, color="red")
clock = axp.text(0.05, 0.93, "", transform=axp.transAxes, fontsize=CLOCK_SIZE)

# -----------------------------------------------------
# Panel de energía
# -----------------------------------------------------
lo = min(K.min(), U.min(), E.min())
hi = max(K.max(), U.max(), E.max())
margen = 0.05 * (hi - lo)
axe.set_xlim(0, tmax)
axe.set_ylim(lo - margen, hi + margen)
axe.set_title("Mechanical Energy as a Function of Time", fontsize=TITLE_SIZE)
axe.set_xlabel("t [s]", fontsize=LABEL_SIZE)
axe.set_ylabel("E [J]", fontsize=LABEL_SIZE)
axe.grid(alpha=0.3)
axe.tick_params(axis="both", labelsize=TICK_SIZE)

# -----------------------------------------------------
# Curvas de energía
# -----------------------------------------------------
lK, = axe.plot([], [], lw=1.1, color="blue", label="K")
lU, = axe.plot([], [], lw=1.1, color="red", label="U")
lE, = axe.plot([], [], lw=1.1, color="green", label="E")
dotE, = axe.plot([], [], "o", ms=3)
axe.legend(loc="upper right", fontsize=LEGEND_SIZE, handlelength=1)

# -----------------------------------------------------
# Animación
# -----------------------------------------------------
def animate(i):
    radius_line.set_data([0, xp[i]],[0, yp[i]])
    line.set_data([xp[i], xb[i]],[yp[i], yb[i]])
    bob.set_data([xb[i]],[yb[i]])
    trace.set_data(xb[:i+1],yb[:i+1])
    clock.set_text(f"t = {t[i]:.2f} s")
    # Energía
    lK.set_data(t[:i+1],K[:i+1])
    lU.set_data(t[:i+1],U[:i+1])
    lE.set_data(t[:i+1],E[:i+1])
    dotE.set_data([t[i]],[E[i]])
    return (radius_line, line, bob, trace, clock, lK, lU, lE, dotE)

# -----------------------------------------------------
# Ejecutar animación
# -----------------------------------------------------
ani = FuncAnimation(fig,animate, frames=range(0, n + 1, STRIDE), 
            interval=STRIDE * dt * 1000 / SPEED, blit=True)
plt.subplots_adjust(hspace=0.35,top=0.93,bottom=0.08)
plt.show()