import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# 1) Reducción a primer orden y solución general
x = sp.symbols('x')
y = sp.Function('y')(x)

# EDO original
ode = sp.Eq(x*sp.diff(y, x, 2) + 2*sp.diff(y, x), 6*x)

# Sustitución v = y'(x)
v = sp.Function('v')(x)
ode_v = sp.Eq(x*sp.diff(v, x) + 2*v, 6*x)
# Forma lineal v' + (2/x) v = 6
ode_v_std = sp.Eq(sp.diff(v, x) + (2/x)*v, 6)

# Resolver para v (Sympy)
C1 = sp.symbols('C1')
sol_v = sp.dsolve(ode_v_std)
# Integrar para y
C2 = sp.symbols('C2')
v_expr = sol_v.rhs
y_expr = sp.integrate(v_expr, (x)) + C2

# Dar formato: y(x) = x^2 - C1/x + C2
A = sp.symbols('A')
y_general = x**2 - A/x + C2  # con A = C1


# 4) IVPs

def solve_ivp_at_1(y1):
    A_val = 0
    C2_val = y1 - 1 + A_val
    y_sol = (x**2 - A_val/x + C2_val).simplify()
    return sp.simplify(y_sol), {"A": A_val, "C2": C2_val}

def solve_ivp_at_0(y0):
    # Regularidad en 0 exige A=0 y C2 = y(0) = y0
    A_val = 0
    C2_val = y0
    y_sol = (x**2 - A_val/x + C2_val).simplify()
    return sp.simplify(y_sol), {"A": A_val, "C2": C2_val}

ivps = {}
ivps["y(1)=2"]  = solve_ivp_at_1(2)
ivps["y(1)=-2"] = solve_ivp_at_1(-2)
ivps["y(1)=1"]  = solve_ivp_at_1(1)
ivps["y(0)=-3"] = solve_ivp_at_0(-3)

# 5) Graficar 
def y_fun_factory(A_val, C2_val):
    # y(x) = x^2 - A/x + C2
    def f(arr):
        if A_val == 0:
            return arr**2 + C2_val
        else:
             return arr**2 - A_val/arr + C2_val
    return f

x_grid = np.linspace(-3, 3, 400)
plt.figure()
labels = []
for name, (y_expr_sol, pars) in ivps.items():
    A_val = float(pars["A"])
    C2_val = float(pars["C2"])
    f = y_fun_factory(A_val, C2_val)
    y_vals = f(x_grid)
    plt.plot(x_grid, y_vals, label=f"{name}: y = x^2 + {C2_val:+g}")

plt.axvline(0, linestyle="--", linewidth=1)
plt.axhline(0, linestyle="--", linewidth=1)
plt.legend()
plt.title("Soluciones con A=0 (sin término 1/x)")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.tight_layout()
plt.savefig("Ejercicio3/soluciones_ivp.png", dpi=150)

# 6) Imprimir un reporte en consola
print("=== Ecuación original ===")
sp.pprint(sp.Eq(x*sp.diff(y, x, 2) + 2*sp.diff(y, x), 6*x))

print("\n=== Reducción a primer orden ===")
sp.pprint(ode_v_std)

print("\n=== Solución para v(x)=y'(x) ===")
print("v(x) =", sp.simplify(sp.dsolve(ode_v_std).rhs))

print("\n=== Solución general para y(x) ===")
print("y(x) = x^2 - A/x + C2")


print("\n=== IVPs (asumiendo A=0) ===")
for name, (y_expr_sol, pars) in ivps.items():
    print(f"{name} -> y(x) = {sp.simplify(y_expr_sol)}   (parámetros: {pars})")

print("\nFigura guardada en: soluciones_ivp.png")

if __name__ == "__main__":
    pass
