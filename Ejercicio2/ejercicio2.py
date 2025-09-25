# ============================================
# Modelación y Simulación - Lab 05
# Ejercicio 2: EDOs de 1er orden (solución + análisis cualitativo)
# Autor: Edwin Ortega 22305, Esteban Zambrano 22119, Juan Pablo SOlis 22102
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp

# ----------------------------
# Utilidades de visualización
# ----------------------------

def slope_field(ax, f, xlim=(-3, 3), ylim=(-3, 3), density=21, normalize=True):
    """
    Dibuja un campo de direcciones y' = f(x, y) en el eje ax.
    - normalize=True dibuja flechas unitarias para enfatizar dirección.
    """
    x = np.linspace(xlim[0], xlim[1], density)
    y = np.linspace(ylim[0], ylim[1], density)
    X, Y = np.meshgrid(x, y)
    F = f(X, Y)

    U = np.ones_like(F)
    V = F.copy()

    if normalize:
        N = np.sqrt(U**2 + V**2)
        # Evitar división por cero
        N[N == 0] = 1.0
        U /= N
        V /= N

    ax.quiver(X, Y, U, V, angles='xy', pivot='mid', scale=25, linewidth=0.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.25)

def plot_ivp_family(ax, f, ic_list, xspan=(-3, 3), **kwargs):
    """
    Dibuja varias soluciones numéricas (IVPs) para y' = f(x, y) con distintas condiciones iniciales.
    ic_list: lista de (x0, y0)
    """
    for (x0, y0) in ic_list:
        # Resolver hacia adelante y hacia atrás para que la curva cruce el plano
        sol1 = solve_ivp(lambda t, y: f(t, y), (x0, xspan[1]), [y0], max_step=0.05, rtol=1e-6, atol=1e-9)
        sol2 = solve_ivp(lambda t, y: f(t, y), (x0, xspan[0]), [y0], max_step=0.05, rtol=1e-6, atol=1e-9)
        ax.plot(sol1.t, sol1.y[0], **kwargs)
        ax.plot(sol2.t, sol2.y[0], **kwargs)

def plot_implicit_family(ax, F, consts, xlim=(-3,3), ylim=(-3,3), levels_color='C1'):
    """
    Grafica curvas de nivel F(x,y) = c para c en consts, via contour.
    """
    xs = np.linspace(xlim[0], xlim[1], 600)
    ys = np.linspace(ylim[0], ylim[1], 600)
    X, Y = np.meshgrid(xs, ys)
    for c in consts:
        Z = F(X, Y) - c
        ax.contour(X, Y, Z, levels=[0], colors=levels_color, linewidths=1.8)

# ----------------------------
# Ecuaciones del ejercicio
# ----------------------------

# 1) y' = -x y  =>  y(x) = C * exp(-x^2/2)
def f1(x, y):
    return -x * y

def y1_explicit(x, C):
    return C * np.exp(-0.5 * x**2)

# 2) y' = x y   =>  y(x) = C * exp(x^2/2)
def f2(x, y):
    return x * y

def y2_explicit(x, C):
    return C * np.exp(0.5 * x**2)

# 3) x dx + y dy = 0  =>  (1/2)x^2 + (1/2)y^2 = C  =>  x^2 + y^2 = K (familia de circunferencias)
# Forma implícita: G(x,y) = x^2 + y^2
def F3(x, y):
    return x**2 + y**2

# 4) y dx + x dy = 0  =>  d(xy)=0 => xy = C (familia hiperbólica simétrica)
def F4(x, y):
    return x * y

# 5) y' = y^2 - y = y(y-1)  =>  separable
#    ∫ dy/(y(y-1)) = ∫ dx  -> ln|(y-1)/y| = x + C  -> y = 1 / (1 + C e^x)
#    + soluciones de equilibrio: y ≡ 0, y ≡ 1
def f5(x, y):
    return y * (y - 1)

def y5_explicit(x, C):
    return 1.0 / (1.0 + C * np.exp(x))  # Forma equivalente a 1 / (1 - K e^x)

# ----------------------------
# Impresión simbólica (opcional pero útil para el informe)
# ----------------------------

def print_symbolic_solutions():
    x = sp.symbols('x')
    y = sp.Function('y')(x)
    print("== Soluciones analíticas (forma general) ==")

    # 1) y' = -x y
    sol1 = sp.dsolve(sp.Eq(sp.diff(y, x), -x*y))
    print("1) y' = -x y  =>", sol1)

    # 2) y' = x y
    sol2 = sp.dsolve(sp.Eq(sp.diff(y, x), x*y))
    print("2) y' = x y   =>", sol2)

    # 3) x dx + y dy = 0  (implícita)
    print("3) x dx + y dy = 0  =>  x^2 + y^2 = C  (implícita; circunferencias)")

    # 4) y dx + x dy = 0  (implícita)
    print("4) y dx + x dy = 0  =>  x*y = C  (implícita; familias hiperbólicas)")

    # 5) y' = y^2 - y
    C = sp.symbols('C')
    print("5) y' = y^2 - y  =>  y(x) = 1 / (1 + C e^x)  +  soluciones de equilibrio y≡0, y≡1")

# ----------------------------
# Main: figuras y reporte rápido
# ----------------------------

def main():
    print_symbolic_solutions()

    # --- Figura 1: y' = -x y ---
    fig, ax = plt.subplots(figsize=(6,5))
    ax.set_title("1) y' = -x y  |  y = C e^{-x^2/2}")
    slope_field(ax, f1, xlim=(-3,3), ylim=(-3,3), density=25, normalize=True)
    X = np.linspace(-3, 3, 400)
    for C in [2.0, 1.0, 0.5, -1.0, -2.0]:
        ax.plot(X, y1_explicit(X, C), lw=2, label=f"C={C}")
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()   # <--- al cerrar esta, pasa a la siguiente

    # --- Figura 2: y' = x y ---
    fig, ax = plt.subplots(figsize=(6,5))
    ax.set_title("2) y' = x y  |  y = C e^{x^2/2}")
    slope_field(ax, f2, xlim=(-2.5,2.5), ylim=(-2.5,2.5), density=25, normalize=True)
    X = np.linspace(-2.5, 2.5, 400)
    for C in [0.5, 1.0, 2.0, -0.5, -1.0]:
        ax.plot(X, y2_explicit(X, C), lw=2, label=f"C={C}")
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()

    # --- Figura 3: circunferencias ---
    fig, ax = plt.subplots(figsize=(6,5))
    ax.set_title("3) x dx + y dy = 0  |  x^2 + y^2 = C")
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.grid(True, alpha=0.25)
    ax.set_xlim(-3,3); ax.set_ylim(-3,3)
    consts3 = [0.5, 1.0, 2.0, 3.0, 4.5]
    plot_implicit_family(ax, F3, consts3, xlim=(-3,3), ylim=(-3,3), levels_color='C2')
    def f3_dir(x,y): 
        with np.errstate(divide='ignore', invalid='ignore'):
            return -x/(y + 1e-12)
    slope_field(ax, f3_dir, xlim=(-3,3), ylim=(-3,3), density=21, normalize=True)
    plt.tight_layout()
    plt.show()

    # --- Figura 4: hipérbolas ---
    fig, ax = plt.subplots(figsize=(6,5))
    ax.set_title("4) y dx + x dy = 0  |  x y = C")
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.grid(True, alpha=0.25)
    ax.set_xlim(-3,3); ax.set_ylim(-3,3)
    consts4 = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
    plot_implicit_family(ax, F4, consts4, xlim=(-3,3), ylim=(-3,3), levels_color='C3')
    def f4_dir(x,y):
        with np.errstate(divide='ignore', invalid='ignore'):
            return -(y)/(x + 1e-12)
    slope_field(ax, f4_dir, xlim=(-3,3), ylim=(-3,3), density=21, normalize=True)
    plt.tight_layout()
    plt.show()

    # --- Figura 5: y' = y^2 - y ---
    fig, ax = plt.subplots(figsize=(6,5))
    ax.set_title("5) y' = y^2 - y  |  y = 1/(1+Ce^x)")
    slope_field(ax, f5, xlim=(-4,4), ylim=(-2,2), density=25, normalize=True)
    X = np.linspace(-4, 4, 600)
    for C in [-2.0, -0.5, 0.5, 2.0]:
        ax.plot(X, y5_explicit(X, C), lw=2, label=f"C={C}")
    ax.axhline(0, color='k', ls='--', lw=1.2, label='y=0 (equilibrio)')
    ax.axhline(1, color='k', ls='-.', lw=1.2, label='y=1 (equilibrio)')
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
