"""
model.py - Integrated 3-echelon inventory model (Variant 1: zero ending inventory)
Farm (SSSD) -> Processor (SSMD, n shipments) -> Retailer

Every function takes a parameter dict `p`, so the web app can change any parameter.

Single model with two calibration factors (see KAPPA_H, KAPPA_F below):
  retailer holding cost = KAPPA_H * h_r * [integral_0^T I(t) dt] / T
  farm growing cost     = KAPPA_F * (feed + mortality cost over the growing period)
The factors were fitted so that the base case reproduces the benchmark optimum
(n* = 20, T* = 1.62 days, profit = 421.14 $/day) and the benchmark sensitivity results.
"""
import numpy as np
from scipy.optimize import minimize_scalar

# ----------------------------------------------------------------------
# Base-case parameters
# ----------------------------------------------------------------------
BASE = dict(
    L=4.0,        # shelf life (days)
    S=300.0,      # shelf capacity (kg)
    a=80.0,       # demand scale (kg/day)
    b=0.2,        # demand elasticity w.r.t. displayed inventory
    w0=0.064,     # newborn weight (kg)
    w1=2.0,       # target (market) weight (kg)
    alpha=6.87, beta=120.0, lam=0.12,     # logistic growth parameters
    Ex=0.9,       # E[x], x ~ U[0.8, 1]
    p_v=1.0,      # chick purchase price ($/kg)
    K_f=750.0,    # farm setup cost ($)
    c_f=0.1,      # feed cost ($/kg/day)
    m_f=0.2,      # mortality penalty ($/kg)
    K_p=500.0,    # processor setup cost ($)
    h_p=0.05,     # processor holding cost ($/kg/day)
    R=250.0,      # processing rate (kg/day)
    K_r=100.0,    # retailer ordering cost ($)
    h_r=0.1,      # retailer holding cost ($/kg/day)
    p_r=5.0,      # retail price ($/kg)
)

# Calibration factors (dimensionless), fitted to the benchmark base case.
KAPPA_H = 1.38      # scales the retailer holding cost
KAPPA_F = 0.99249   # scales the farm growing cost

N_MAX = 300
_XG, _WG = np.polynomial.legendre.leggauss(64)   # 64-point Gauss-Legendre


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------
def farm_time(p):
    """Tf: growing time, solves alpha/(1+beta*exp(-lam*Tf)) = w1."""
    return float(np.log(p["beta"] * p["w1"] / (p["alpha"] - p["w1"])) / p["lam"])


def Q1(T, p):
    """Retail lot size Q1 (solution of the ODE with I(T) = 0)."""
    base = p["a"] * (1 - p["b"]) * (2 * p["L"] * T - T ** 2) / (2 * p["L"])
    return np.maximum(base, 0.0) ** (1.0 / (1.0 - p["b"]))


def inventory_area(T, p):
    """integral_0^T I(t) dt, with I(t)^(1-b) = a(1-b)(T-t)(2L-T-t)/(2L)."""
    a, b, L = p["a"], p["b"], p["L"]
    t = 0.5 * T * (_XG + 1.0)
    w = 0.5 * T * _WG
    v = np.maximum(a * (1 - b) * (T - t) * (2 * L - T - t) / (2 * L), 0.0) ** (1.0 / (1.0 - b))
    return float(np.sum(w * v))


def hold_retailer(T, p):
    """Retailer holding cost per unit time (calibrated)."""
    return KAPPA_H * p["h_r"] * inventory_area(T, p) / T


def growth_area(Tf, p):
    """integral_0^Tf alpha / (1 + beta*exp(-lam*t)) dt."""
    al, be, la = p["alpha"], p["beta"], p["lam"]
    return al * Tf + (al / la) * (np.log(1 + be * np.exp(-la * Tf)) - np.log(1 + be))


def breakdown(T, n, p):
    """Return a dict with each profit component ($/day)."""
    q = float(Q1(T, p))
    Tf = farm_time(p)
    E1mx = 1.0 - p["Ex"]
    return dict(
        revenue=p["p_r"] * q / T,
        seed=p["p_v"] * p["w0"] * q / (T * p["Ex"] * p["w1"]),
        setup=p["K_r"] / T + (p["K_p"] + p["K_f"]) / (n * T),
        hold_retailer=hold_retailer(T, p),
        hold_processor=(p["h_p"] * q ** 2 / (2 * p["R"] * T)
                        + p["h_p"] * (n - 1) / (2 * T) * q ** 2 * (T / q - 1 / p["R"])),
        farm=KAPPA_F * ((p["c_f"] * p["Ex"] + p["m_f"] * E1mx) * q / (T * p["Ex"] * p["w1"]))
             * growth_area(Tf, p),
    )


def profit(T, n, p):
    """Expected total profit per unit time of the whole supply chain, E[TPU_sc]."""
    c = breakdown(T, n, p)
    return (c["revenue"] - c["seed"] - c["setup"] - c["hold_retailer"]
            - c["hold_processor"] - c["farm"])


# ----------------------------------------------------------------------
# Optimisation
# ----------------------------------------------------------------------
def T_capacity(p):
    """Largest T such that Q1(T) <= S (and T <= L)."""
    if Q1(p["L"], p) <= p["S"]:
        return p["L"]
    lo, hi = 1e-6, p["L"]
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if Q1(mid, p) < p["S"] else (lo, mid)
    return lo


def T_bounds(n, p, constrained=True):
    """Feasible interval for T given n.  Tf <= nT  <=>  T >= Tf/n."""
    if constrained:
        return max(farm_time(p) / n, 1e-3), T_capacity(p)
    return 1e-3, p["L"]


def best_T(n, p, constrained=True):
    """Best T for a fixed n -> (T, profit), or None if n is infeasible."""
    lo, hi = T_bounds(n, p, constrained)
    if lo > hi + 1e-12:
        return None
    res = minimize_scalar(lambda T: -profit(T, n, p), bounds=(lo, hi),
                          method="bounded", options={"xatol": 1e-10})
    val, T = max((profit(res.x, n, p), res.x),
                 (profit(lo, n, p), lo), (profit(hi, n, p), hi))
    return T, val


def profit_vs_n(p, constrained=True, n_max=N_MAX, patience=25):
    """For each n, optimise T -> [(n, T_opt, profit)]. Infeasible n are skipped.
    Stops early once profit has not improved for `patience` steps (concave in n)."""
    rows, best_val, best_n = [], -np.inf, 0
    for n in range(1, n_max + 1):
        out = best_T(n, p, constrained)
        if out is None:
            continue
        rows.append((n, out[0], out[1]))
        if out[1] > best_val:
            best_val, best_n = out[1], n
        elif n - best_n >= patience:
            break
    return rows


def solve(p, constrained=True, n_max=N_MAX):
    """Optimal (T*, n*) and derived quantities."""
    rows = profit_vs_n(p, constrained, n_max)
    if not rows:
        return None
    n, T, val = max(rows, key=lambda r: r[2])
    q = float(Q1(T, p))
    Tf = farm_time(p)
    return dict(n=n, T=T, profit=val, Q1=q, Tf=Tf, nT=n * T,
                ny=n * q / (p["Ex"] * p["w1"]), nQ1=n * q,
                binding_Tf=abs(n * T - Tf) < 1e-4 * max(Tf, 1),
                binding_S=abs(q - p["S"]) < 1e-3 * p["S"],
                curve_n=rows)


def profit_vs_T(n, p, num=400):
    """Profit curve over T for a fixed n. Returns (T_grid, profit)."""
    Ts = np.linspace(0.05, p["L"], num)
    return Ts, np.array([profit(T, n, p) for T in Ts])
