# ============================================
# Modelación y Simulación - Lab 05
# Ejercicio 4: Campo de direcciones, solución de PVI y puntos de equilibrio
# Autor: Edwin Ortega 22305, Esteban Zambrano 22119, Juan Pablo Solis 22102
# ============================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root


def P(x, y):
    return x - 3*y - 3*(x**2 - y**2) + 3*x*y

def Q(x, y):
    return 2*x - y + 3*(x**2 - y**2) + 2*x*y

def f(x, y):
    """ RHS de la EDO: dy/dx = P/Q. Evita división por cero en ploteo. """
    den = Q(x, y)
    return P(x, y) / den

# (a) Campo de direcciones


# Rango del plano para visualizar
x_min, x_max = -3.0, 3.0
y_min, y_max = -3.0, 3.0

# Malla para slope field
nx, ny = 31, 31
X = np.linspace(x_min, x_max, nx)
Y = np.linspace(y_min, y_max, ny)
XX, YY = np.meshgrid(X, Y)

# Slope field como quiver con vectores (1, f). Se normalizan.
FF = np.zeros_like(XX, dtype=float)
den = Q(XX, YY)
mask = np.abs(den) > 1e-8           # evitamos división por ~0
FF[mask] = P(XX[mask], YY[mask]) / den[mask]
# Componentes normalizadas
U = 1.0 / np.sqrt(1.0 + FF**2)
V = FF / np.sqrt(1.0 + FF**2)

fig, ax = plt.subplots(figsize=(7.2, 6))
ax.quiver(XX, YY, U, V, color='0.35', pivot='mid', angles='xy', scale=25)

# (b) Solución del PVI
x0 = 1.5
y0 = 0.0

def rhs(x, y):
    return f(x, y[0])

# Evento para detener integración si Q(x,y)=0 (singularidad de dy/dx)
def event_Q_zero(x, y):
    return Q(x, y[0])
event_Q_zero.terminal = True
event_Q_zero.direction = 0.0

# Integramos hacia adelante y hacia atrás en x
x_to_right = (x0, x_max)
x_to_left  = (x0, x_min)

sol_right = solve_ivp(rhs, x_to_right, [y0],
                      dense_output=True, max_step=0.05,
                      events=event_Q_zero, rtol=1e-8, atol=1e-10)

sol_left  = solve_ivp(rhs, x_to_left,  [y0],
                      dense_output=True, max_step=0.05,
                      events=event_Q_zero, rtol=1e-8, atol=1e-10)

# Construimos curva completa
xs_r = np.linspace(sol_right.t[0], sol_right.t[-1], 800)
ys_r = sol_right.sol(xs_r)[0]
xs_l = np.linspace(sol_left.t[-1], sol_left.t[0], 800)
ys_l = sol_left.sol(xs_l)[0]

# Dibujamos solución
ax.plot(xs_l, ys_l, 'C1', lw=2.2, label='Sol. PVI (hacia la izquierda)')
ax.plot(xs_r, ys_r, 'C0', lw=2.2, label='Sol. PVI (hacia la derecha)')
ax.plot([x0], [y0], 'ko', ms=6, label='Condición inicial (1.5, 0)')

# (c) Puntos de equilibrio
# Se interpreta la EDO como sistema:
#   dx/dt = Q(x,y),  dy/dt = P(x,y)
# Puntos de equilibrio: P(x,y)=0 y Q(x,y)=0 simultáneamente.

def F(v):
    x, y = v
    return np.array([P(x, y), Q(x, y)])

# Buscamos raíces desde múltiples semillas en una rejilla
seeds_x = np.linspace(-3, 3, 13)
seeds_y = np.linspace(-3, 3, 13)

roots = []
for sx in seeds_x:
    for sy in seeds_y:
        sol = root(F, x0=np.array([sx, sy]), method='hybr')
        if sol.success and np.linalg.norm(F(sol.x)) < 1e-8:
            roots.append(sol.x)

# Quitamos duplicados cercanos
eq_pts = []
tol = 1e-4
for r in roots:
    if not any(np.linalg.norm(r - e) < tol for e in eq_pts):
        eq_pts.append(r)

# Mostramos y dibujamos los puntos de equilibrio
for i, (xe, ye) in enumerate(eq_pts, 1):
    ax.plot(xe, ye, 'r*', ms=10)
    print(f"Punto de equilibrio #{i}: (x, y) = ({xe:.6f}, {ye:.6f})")


# Formato final del gráfico
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Campo de direcciones y solución del PVI  y(1.5)=0')
ax.grid(True, alpha=0.25)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('Ejercicio4/campo_y_solucion.png', dpi=150)
plt.show()
print("\nFigura guardada como: Ejercicio4/campo_y_solucion.png")
