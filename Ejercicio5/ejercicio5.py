# ============================================
# Modelación y Simulación - Lab 05
# Ejercicio 5: Modelo poblacional  P' = 0.0004 P^2 - 0.06 P
# Autor: Edwin Ortega 22305, Esteban Zambrano 22119, Juan Pablo SOlis 22102
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp

# Parámetros del modelo
a = 0.0004
b = 0.06

# f(P) y análisis básico
def fP(P):
    return a*P**2 - b*P

def dfP(P):
    return 2*a*P - b

# Solución analítica (separable):
# ln((aP - b)/P) = b t + C  ->  P(t) =  b / [ a - (a - b/P0) e^{b t} ]
def P_analytic(t, P0):
    denom = a - (a - b/P0)*np.exp(b*t)
    return b / denom

# Tiempo de blow-up si aplica (P0> b/a = 150); si no, devuelve None
def blowup_time(P0):
    if P0 <= b/a:
        return None
    # a - (a - b/P0) e^{b t} = 0  -> e^{b t} = a/(a - b/P0)
    val = a / (a - b/P0)
    if val <= 0:
        return None
    return (1.0/b) * np.log(val)

def main():
    print("== Ejercicio 5:  P' = a P^2 - b P,  con a=0.0004, b=0.06 ==")
    print("Equilibrios: P*=0 y P*=b/a=150")
    print(f"Estabilidad: f'(0)={dfP(0):.3f} < 0 -> estable; f'(150)={dfP(150):.3f} > 0 -> inestable")
    print("Solución analítica:  P(t) = b / [ a - (a - b/P0) e^{b t} ]")
    print()

    # -----------------------------
    # 1) Diagrama de fase (f(P) vs P) y estabilidad
    # -----------------------------
    Ps = np.linspace(-50, 350, 800)
    fig, ax = plt.subplots(figsize=(6.5,4.8))
    ax.plot(Ps, fP(Ps), lw=2)
    ax.axhline(0, color='k', lw=1)
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.axvline(b/a, color='k', ls='--', lw=1, label='P*=150')
    ax.set_title("Diagrama de fase: f(P) = aP² - bP")
    ax.set_xlabel("P")
    ax.set_ylabel("P' = f(P)")
    ax.grid(True, alpha=0.25)
    # Flechitas de flujo 1D
    for P0 in [25, 75, 125, 175, 250]:
        d = fP(P0)
        ax.annotate("", xy=(P0 + np.sign(d)*15, 0.2*np.sign(d)), xytext=(P0, 0.2*np.sign(d)),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 2) Campo direccional en el plano (t,P)
    # -----------------------------
    tmin, tmax = 0.0, 30.0
    Pmin, Pmax = -10.0, 320.0
    t = np.linspace(tmin, tmax, 26)
    P = np.linspace(Pmin, Pmax, 26)
    T, PP = np.meshgrid(t, P)
    U = np.ones_like(T)
    V = fP(PP)  # P' no depende de t (autónoma)

    N = np.sqrt(U**2 + V**2)
    N[N == 0] = 1.0
    U /= N; V /= N

    fig, ax = plt.subplots(figsize=(7,5))
    ax.quiver(T, PP, U, V, angles='xy', scale=30, width=0.003)
    ax.axhline(0, color='k', ls='--', lw=1.2, label='Equilibrio P=0 (estable)')
    ax.axhline(b/a, color='k', ls='-.', lw=1.2, label='Equilibrio P=150 (inestable)')
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(Pmin, Pmax)
    ax.set_title("Campo direccional en (t, P) para P' = 0.0004 P² - 0.06 P")
    ax.set_xlabel("t")
    ax.set_ylabel("P")
    ax.grid(True, alpha=0.25)
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 3) Soluciones para P0=100 y P0=200 (analítica + numérica)
    # -----------------------------
    ICs = [100.0, 200.0]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.set_title("Soluciones P(t) con P(0)=100 y P(0)=200")
    ax.set_xlabel("t")
    ax.set_ylabel("P(t)")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color='k', ls='--', lw=1.0, label='P=0')
    ax.axhline(b/a, color='k', ls='-.', lw=1.0, label='P=150')

    colors = ["C0", "C3"]
    for k, P0 in enumerate(ICs):
        # Analítica
        t_grid = np.linspace(0, tmax, 4000)
        # Evitar dividir cerca del polo (si lo hay)
        denom = a - (a - b/P0)*np.exp(b*t_grid)
        mask = np.abs(denom) > 1e-8
        ax.plot(t_grid[mask], (b/denom)[mask], lw=2.2, color=colors[k],
                label=f"Analítica  P(0)={int(P0)}")

        # Numérica (final seguro antes del blow-up)
        t_end = tmax
        t_star = blowup_time(P0)
        if t_star is not None:
            t_end = min(tmax, t_star - 0.1)  # nos quedamos antes del polo
        sol = solve_ivp(lambda tt, PP: fP(PP), (0, t_end), [P0],
                        max_step=0.02, rtol=1e-9, atol=1e-12)
        ax.plot(sol.t, sol.y[0], '--', color=colors[k], lw=1.6,
                label=f"Numérica  P(0)={int(P0)}")

        # Reporte en consola
        if t_star is None:
            print(f"P0={P0:.0f}: sin blow-up; la trayectoria desciende hacia 0 (equilibrio estable).")
        else:
            print(f"P0={P0:.0f}: blow-up en t* ≈ {t_star:.3f} (P(t)→∞); el equilibrio P=150 es inestable.")

    ax.set_xlim(0, tmax)
    ax.set_ylim(Pmin, Pmax)
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 4) Resumen simbólico (útil para el informe)
    # -----------------------------
    t = sp.symbols('t', real=True)
    P = sp.Function('P')(t)
    sol = sp.dsolve(sp.Eq(sp.diff(P, t), a*P**2 - b*P))
    print("\n== Solución simbólica (forma general) ==")
    print(sol)
    print("\nInterpretación:")
    print("* Equilibrio 0: estable (las trayectorias con 0 < P0 < 150 tienden a 0).")
    print("* Equilibrio 150: inestable (cualquier P0 > 150 diverge; P0 < 150 se aleja).")
    print("* Para P0=200, el tiempo de blow-up t* =", f"{blowup_time(200):.3f}", "aprox.")
    print("* Para P0=100, la solución decrece monótonamente hacia 0 sin blow-up.")

if __name__ == "__main__":
    main()
