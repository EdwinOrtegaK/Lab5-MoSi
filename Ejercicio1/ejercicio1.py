# ============================================
# Modelación y Simulación - Lab 05
# Ejercicio 1: Campo de direcciones para y' = f(x,y)
# Autor: Edwin Ortega 22305, Esteban Zambrano 22119, Juan Pablo SOlis 22102
# ============================================

import os
import numpy as np
import matplotlib.pyplot as plt

def _build_F(f, X, Y, unit=False, eps=1e-12):
    """
    Construye el campo F=(U,V) para y'=f(x,y).
    - Si unit=True, devuelve el campo unitario N = (1,f)/||(1,f)||.
    - Evita divisiones por cero con 'eps'.
    """
    U = np.ones_like(X, dtype=float)
    V = f(X, Y).astype(float)

    if unit:
        N = np.sqrt(U*U + V*V)
        N = np.where(N < eps, 1.0, N)  # evita 0
        U = U / N
        V = V / N

    # Limpieza de NaN/Inf (por si f tiene singularidades)
    U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    return U, V

def plot_direction_field(
    f,
    xmin, xmax, ymin, ymax,
    xstep, ystep,
    *,
    unit=True,
    quiver_scale=25,
    quiver_width=0.002,
    arrows_every=1,
    show_stream=False,
    stream_density=1.2,
    stream_lw=1.2,
    title=None,
    xlabel="x",
    ylabel="y",
    grid_alpha=0.25,
):
    """
    Dibuja el campo de direcciones del ODE de 1er orden y' = f(x,y).

    Parámetros obligatorios:
    - f      : función vectorizada que acepte arreglos (X,Y) y retorne f(X,Y)
    - xmin,xmax,ymin,ymax : límites de la ventana
    - xstep, ystep : separación del grid en cada eje

    Retorna:
    - (fig, ax)
    """
    # Construir grilla
    xs = np.arange(xmin, xmax + 1e-12, xstep, dtype=float)
    ys = np.arange(ymin, ymax + 1e-12, ystep, dtype=float)
    X, Y = np.meshgrid(xs, ys)

    # Campo (U,V) = (1, f(x,y))
    U, V = _build_F(f, X, Y, unit=unit)

    # Submuestreo para quiver si hace falta
    slc = (slice(None, None, arrows_every), slice(None, None, arrows_every))

    # Figura
    fig, ax = plt.subplots(figsize=(7,5))
    q = ax.quiver(
        X[slc], Y[slc], U[slc], V[slc],
        angles="xy",
        pivot="mid",
        scale=quiver_scale,
        width=quiver_width
    )

    if show_stream:
        # Para streamplot es usual usar el campo "continuo" sin submuestreo
        ax.streamplot(
            X, Y, U, V,
            density=stream_density,
            linewidth=stream_lw,
            arrowsize=1.0,
            minlength=0.05
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=grid_alpha)

    # Leyenda mínima del modo
    mode = "unitario N" if unit else "original F"
    ax.text(
        0.02, 0.98, f"Campo {mode}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.8)
    )

    return fig, ax

# ------------------------------------------
# Ejemplos de uso (ilustrar dos campos)
# ------------------------------------------
OUTPUT_DIR = "Ejercicio1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def ejemplo_1():
    def f1(X, Y):
        return -X*Y

    fig, ax = plot_direction_field(
        f1, xmin=-3, xmax=3, ymin=-3, ymax=3,
        xstep=0.3, ystep=0.3,
        unit=True,
        show_stream=True,
        title="Ejemplo 1:  y' = -x y"
    )
    path = os.path.join(OUTPUT_DIR, "ejemplo1.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)   # cerrar para no abrir ventana
    print(f"Guardado: {path}")

def ejemplo_2():
    def f2(X, Y):
        return Y*(Y-1.0)

    fig, ax = plot_direction_field(
        f2, xmin=-4, xmax=4, ymin=-2, ymax=2,
        xstep=0.25, ystep=0.25,
        unit=True,
        show_stream=True,
        title="Ejemplo 2:  y' = y(y-1)"
    )
    # Nullclines
    ax.axhline(0, color="k", ls="--", lw=1.0)
    ax.axhline(1, color="k", ls="-.", lw=1.0)

    path = os.path.join(OUTPUT_DIR, "ejemplo2.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {path}")

if __name__ == "__main__":
    ejemplo_1()
    ejemplo_2()