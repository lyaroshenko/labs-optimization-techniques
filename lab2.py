import numpy as np
import matplotlib.pyplot as plt

#  ROSENBROCK FUNCTION

def f(x):
    return 100.0 * (x[1] - x[0]**2)**2 + (1.0 - x[0])**2


def grad_f(x):
    u      = x[1] - x[0]**2
    df_dx1 = -400.0 * x[0] * u  - 2.0 * (1.0 - x[0])
    df_dx2 =  200.0 * u
    return np.array([df_dx1, df_dx2])


def hess_f(x):
    h11 = 1200.0 * x[0]**2 - 400.0 * x[1] + 2.0
    h12 = -400.0 * x[0]
    h22 = 200.0
    return np.array([[h11, h12],
                     [h12, h22]])


#  PART 1 — NEWTON'S METHOD

def newton(f, grad_f, hess_f, x0, eps=1e-6, max_iter=10000):
    x = np.array(x0, dtype=float)

    positions    = [x.copy()]
    fvals        = [f(x)]
    directions   = []
    hessians     = []
    inv_hessians = []
    epsilons     = []

    for k in range(max_iter):

        g = grad_f(x)
        H = hess_f(x)

        d = np.linalg.solve(H, -g)

        H_inv = np.linalg.inv(H)

        x_new = x + d
        eps_k = np.linalg.norm(d)

        directions.append(d.copy())
        hessians.append(H.copy())
        inv_hessians.append(H_inv.copy())
        epsilons.append(eps_k)

        x = x_new
        positions.append(x.copy())
        fvals.append(f(x))

        if abs(fvals[-1] - fvals[-2]) < eps:
            break

    return (np.array(positions), np.array(fvals),
            directions, hessians, inv_hessians,
            np.array(epsilons), k + 1)


#  PART 2 — QUASI-NEWTON: BFGS

def quasi_newton(f, grad_f, hess_f, x0, eps=1e-6, max_iter=10000):
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)

    positions       = [x.copy()]
    fvals           = [f(x)]
    directions      = []
    H_list          = []
    epsilons        = []
    hess_diff_norms = []

    g = grad_f(x)

    for k in range(max_iter):

        d = -H @ g

        x_new = x + d
        g_new = grad_f(x_new)

        s     = x_new - x
        y     = g_new - g
        eps_k = np.linalg.norm(s)

        directions.append(d.copy())
        H_list.append(H.copy())
        epsilons.append(eps_k)

        H_true_inv = np.linalg.inv(hess_f(x))
        diff_norm  = np.linalg.norm(H_true_inv - H, 2)
        hess_diff_norms.append(diff_norm)
        print(f"  Iter {k+1:5d}:  f = {f(x):12.4f}   "
              f"‖(∇²f)⁻¹ − Hₖ‖ = {diff_norm:.6f}")

        x = x_new
        g = g_new
        positions.append(x.copy())
        fvals.append(f(x))

        if abs(fvals[-1] - fvals[-2]) < eps:
            break

        sy = float(s @ y)

        if sy < 1e-10:
            print(f"    [iter {k+1}: curvature condition violated — update skipped]")
            continue

        rho = 1.0 / sy

        I_n = np.eye(n)
        A   = I_n - rho * np.outer(s, y)
        B   = I_n - rho * np.outer(y, s)

        H = A @ H @ B + rho * np.outer(s, s)

    return (np.array(positions), np.array(fvals),
            directions, H_list,
            np.array(epsilons), hess_diff_norms, k + 1)


#  RUN BOTH METHODS FROM BOTH STARTING POINTS

x0_a = np.array([ 2.0,  4.0])
x0_b = np.array([-2.0, 10.0])

print("═" * 65)
print("  PART 1: NEWTON'S METHOD")
print("═" * 65)
pos_n_a,fv_n_a,dir_n_a,H_n_a,Hinv_n_a,eps_n_a,it_n_a = newton(f,grad_f,hess_f,x0_a)
pos_n_b,fv_n_b,dir_n_b,H_n_b,Hinv_n_b,eps_n_b,it_n_b = newton(f,grad_f,hess_f,x0_b)
print(f"  x⁰=(2,4)  : {it_n_a} iterations,  "
      f"x*≈{pos_n_a[-1]},  f*≈{fv_n_a[-1]:.2e}")
print(f"  x⁰=(-2,10): {it_n_b} iterations,  "
      f"x*≈{pos_n_b[-1]},  f*≈{fv_n_b[-1]:.2e}")

print("\n" + "═" * 65)
print("  PART 2: BFGS  —  x⁰ = (2, 4)")
print("═" * 65)
pos_q_a,fv_q_a,dir_q_a,H_q_a,eps_q_a,hdiff_q_a,it_q_a = quasi_newton(f,grad_f,hess_f,x0_a)
print(f"\n  → {it_q_a} iterations,  x*≈{pos_q_a[-1]},  f*≈{fv_q_a[-1]:.2e}")

print("\n" + "═" * 65)
print("  PART 2: BFGS  —  x⁰ = (-2, 10)")
print("═" * 65)
pos_q_b,fv_q_b,dir_q_b,H_q_b,eps_q_b,hdiff_q_b,it_q_b = quasi_newton(f,grad_f,hess_f,x0_b)
print(f"\n  → {it_q_b} iterations,  x*≈{pos_q_b[-1]},  f*≈{fv_q_b[-1]:.2e}")


#  FIGURE 1 — f(xₖ) vs iteration (one figure per method, both starts overlaid)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Objective function value vs iteration", fontsize=13, fontweight='bold')

for ax, (fv_a, fv_b, title) in zip(axes,
        [(fv_n_a, fv_n_b, "Newton's Method"),
         (fv_q_a, fv_q_b, "BFGS Quasi-Newton")]):
    ax.semilogy(fv_a + 1e-30, 'o-', color='steelblue',  linewidth=2, ms=4,
                label=r"$x^0=(2,4)$")
    ax.semilogy(fv_b + 1e-30, 's--',color='darkorange', linewidth=2, ms=4,
                label=r"$x^0=(-2,10)$")
    ax.set_xlabel("Iteration k"); ax.set_ylabel("f(xₖ) [log scale]")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig("C:/Users/zhuko/PycharmProjects/labs-optimization-techniques", dpi=150, bbox_inches='tight')
plt.show()

#  FIGURE 2 — lg εₖ vs k  (convergence order)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(r"Convergence order: $\lg\,\varepsilon_k$ vs $k$",
             fontsize=13, fontweight='bold')

for ax, (eps_a, eps_b, title) in zip(axes,
        [(eps_n_a, eps_n_b, "Newton's Method"),
         (eps_q_a, eps_q_b, "BFGS Quasi-Newton")]):
    with np.errstate(divide='ignore'):
        ax.plot(np.log10(eps_a+1e-300), 'o-', color='steelblue',  lw=2, ms=4,
                label=r"$x^0=(2,4)$")
        ax.plot(np.log10(eps_b+1e-300), 's--',color='darkorange', lw=2, ms=4,
                label=r"$x^0=(-2,10)$")
    ax.set_xlabel("Iteration k"); ax.set_ylabel(r"$\lg(\varepsilon_k)$")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig("C:/Users/zhuko/PycharmProjects/labs-optimization-techniques", dpi=150, bbox_inches='tight')
plt.show()

#  FIGURE 3 — BFGS Hessian approximation quality

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(r"BFGS: $\|(\nabla^2 f_k)^{-1} - H_k\|_2$ at each iteration",
             fontsize=13, fontweight='bold')

for ax, (hd, lab, col) in zip(axes,
        [(hdiff_q_a, r"$x^0=(2,4)$",   'steelblue'),
         (hdiff_q_b, r"$x^0=(-2,10)$", 'darkorange')]):
    ax.semilogy(hd, 'o-', color=col, lw=2, ms=4)
    ax.set_title(f"Start {lab}")
    ax.set_xlabel("Iteration k")
    ax.set_ylabel(r"$\|(\nabla^2 f_k)^{-1}-H_k\|_2$ [log scale]")
    ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig("C:/Users/zhuko/PycharmProjects/labs-optimization-techniques", dpi=150, bbox_inches='tight')
plt.show()

#  SUMMARY

print("\n" + "═"*65)
print("  FINAL SUMMARY")
print("═"*65)
rows = [("Newton","(2,4)",   it_n_a, fv_n_a[-1]),
        ("Newton","(-2,10)", it_n_b, fv_n_b[-1]),
        ("BFGS",  "(2,4)",   it_q_a, fv_q_a[-1]),
        ("BFGS",  "(-2,10)", it_q_b, fv_q_b[-1])]
print(f"  {'Method':<8} {'Start':<10} {'Iters':>8}  {'f* (final)':>14}")
print("  " + "-"*44)
for m,s,it,fval in rows:
    print(f"  {m:<8} {s:<10} {it:>8}  {fval:>14.4e}")
print("═"*65)

print("""
DISCUSSION ANSWERS
──────────────────
1. Stopping condition:
   Both methods stop when |f(xₖ₊₁) − f(xₖ)| < ε = 10⁻⁶ OR iteration > 10 000.

2. Newton on a positive-definite quadratic:
   The Hessian of a quadratic is CONSTANT (independent of x), so Newton finds the
   exact minimum in EXACTLY 1 ITERATION from any starting point.

3. Disadvantages of Newton + fixes:
   - Cost: O(n²) Hessian entries, O(n³) solve per iteration.
   - The Hessian may be indefinite far from the minimum → d is not a descent dir.
   Fixes: modified Newton (regularise H with λI), trust regions, quasi-Newton.

4. Main idea of quasi-Newton:
   Maintain a cheap approximation Hₖ ≈ (∇²f)⁻¹ updated only with gradients.
   BFGS is the best-known: its rank-2 update satisfies the secant equation
   and preserves positive-definiteness when yₖᵀsₖ > 0.

5. Why BFGS ≠ true inverse Hessian:
   Each update imposes only ONE constraint (the secant equation).  In 2D that
   leaves 2 free parameters per step — BFGS picks the "closest" positive-definite
   solution, which need not equal the true inverse Hessian.

6. Is BFGS always positive definite?
   YES if yₖᵀsₖ > 0 at every step (guaranteed by Wolfe line search).
   With unit step length (as here), the curvature condition can be violated and
   the update must be skipped, so convergence is not guaranteed.

7. Newton vs BFGS iteration count:
   Newton converges in very few iterations (3–5 here) because it uses exact 2nd
   order information.  BFGS uses approximate 2nd order information, so it may
   need more iterations but each step is cheaper (no linear system).
   Without line search on Rosenbrock both are limited by the unit-step constraint.
""")