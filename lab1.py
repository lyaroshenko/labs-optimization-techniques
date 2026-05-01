import numpy as np
import matplotlib.pyplot as plt
import time

# ASSIGNMENT 1: Rosenbrock Function
# The minimum is at point (1, 1) where f = 0.
def calculate_objective_rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def compute_gradient_rosenbrock(x):
    # The gradient tells us the SLOPE at our current position
    # It's a vector pointing UPHILL in both the x0 and x1 directions
    # We computed these formulas using calculus (derivative of the function above)

    # How fast f changes as we move in the x0 direction
    df_dx1 = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])

    # How fast f changes as we move in the x1 direction
    df_dx2 = 200 * (x[1] - x[0] ** 2)

    # Return both slopes as one vector — this points uphill
    return np.array([df_dx1, df_dx2])


def gradient_descent_rosenbrock(x0, gamma, max_iter=10000, tol=1e-4):
    # x0    = starting position (where we begin our descent)
    # gamma = step size (how far we move each iteration)
    # max_iter = maximum number of steps allowed
    # tol   = if the slope is smaller than this, we're close enough to minimum

    x = x0.copy()  # start at the given starting point

    # Record the function value at our starting point (for plotting later)
    history = [calculate_objective_rosenbrock(x)]

    start = time.time()  # start the clock

    for k in range(1, max_iter + 1):

        # STEP 1: Compute the gradient (which direction is uphill?)
        grad = compute_gradient_rosenbrock(x)

        # STEP 2: Compute how steep the slope is (length of the gradient vector)
        grad_norm = np.linalg.norm(grad)

        # Safety check: if values exploded to infinity, stop
        if not np.isfinite(grad_norm):
            break

        # STEP 3: Check if we've reached the minimum (slope is nearly flat)
        if grad_norm < tol:
            break  # slope is tiny → we're at the bottom → stop

        # STEP 4: Move in the OPPOSITE direction of the gradient (downhill)
        # This is the core of gradient descent:
        #   new position = old position - step_size * uphill_direction
        x_new = x - gamma * grad

        # Safety check: if new position is invalid (infinity/NaN), stop
        if not np.all(np.isfinite(x_new)):
            break

        x = x_new  # accept the new position

        # Record the new function value (so we can plot convergence)
        history.append(calculate_objective_rosenbrock(x))

    elapsed = time.time() - start  # how long did it take?
    return history, k, elapsed, x


print("ASSIGNMENT 1: Rosenbrock Function")

x0 = np.array([-2.0, 2.0])
gammas = [0.1, 0.01, 0.001]
results_1 = {}

for gamma in gammas:
    history, iters, elapsed, x_final = gradient_descent_rosenbrock(x0, gamma)
    results_1[gamma] = (history, iters, elapsed, x_final)
    print(f"\nStep size γ = {gamma}:")
    print(f"  Iterations     : {iters}")
    print(f"  Time (s)       : {elapsed:.6f}")
    print(f"  Final x        : [{x_final[0]:.6f}, {x_final[1]:.6f}]")
    print(f"  Final f(x)     : {history[-1]:.6e}")
    converged = np.linalg.norm(compute_gradient_rosenbrock(x_final)) < 1e-4
    print(f"  Converged      : {converged}")

fig1, ax1 = plt.subplots(figsize=(10, 6))
for gamma in gammas:
    history = results_1[gamma][0]
    ax1.semilogy(history, label=f"γ = {gamma}")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("f(x)  [log scale]")
ax1.set_title("Assignment 1 – Rosenbrock: Objective vs Iteration")
ax1.legend()
ax1.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("assignment1_rosenbrock.png", dpi=150)
plt.show()
print("\nPlot saved: assignment1_rosenbrock.png")

# ASSIGNMENT 2: Least Squares Estimation

# GOAL: We have a system  A*x ≈ b
#   A = known matrix (100 rows × 10 columns)
#   b = known measurements (with some noise added)
#   x = UNKNOWN vector we want to find (10 values)
#
# We minimize the MSE (Mean Squared Error):
#   f(x) = (1/2m) * ||A*x - b||²
#         = average of (prediction - actual)²
#
# This is CONVEX and SMOOTH → gradient descent is guaranteed to converge
# The gradient is: ∇f(x) = (1/m) * Aᵀ(Ax - b)
#
# The safe step size is γ = 1/β  where β = largest eigenvalue of AᵀA/m
def calculate_objective_ls(x, A, b):
    # Compute the MSE: how wrong is our current guess x?
    m = len(b)               # number of measurements
    r = A @ x - b            # residual = prediction minus actual (the errors)
    return (1 / (2 * m)) * np.dot(r, r)   # average squared error

def compute_gradient_ls(x, A, b):
    # Gradient of MSE: tells us which direction to step to reduce the error
    # Formula comes from calculus: derivative of (1/2m)||Ax-b||²
    m = len(b)
    return (1 / m) * A.T @ (A @ x - b)
    # A.T @ something = matrix multiply A-transposed by something


def gradient_descent_ls(x0, A, b, gamma, max_iter=50):
    # Standard gradient descent loop — same idea as Assignment 1
    # but applied to the Least Squares problem

    x = x0.copy()
    history = [calculate_objective_ls(x, A, b)]  # record starting error
    start = time.time()

    for _ in range(max_iter):
        grad = compute_gradient_ls(x, A, b)  # compute slope
        x = x - gamma * grad  # step downhill
        history.append(calculate_objective_ls(x, A, b))  # record new error

    elapsed = time.time() - start
    return history, elapsed, x

print("ASSIGNMENT 2: Least Squares Estimation")

np.random.seed(42)
m, n = 100, 10           # 100 measurements, 10 unknowns
A = np.random.randn(m, n) # random matrix
x_true = np.random.randn(n) # the TRUE answer (we pretend not to know this)
b = A @ x_true + 0.1 * np.random.randn(m)   # noisy measurements
x0_ls = np.zeros(n) # start with all zeros as our initial guess

beta = (1 / m) * np.linalg.norm(A, 2)**2
print(f"\nSmoothness constant β = {beta:.4f}")
print(f"Step 1/β = {1/beta:.6f}")

x_norm = np.linalg.norm(x0_ls)
AtA = A.T @ A
Atb = A.T @ b
L = (1 / m) * (np.linalg.norm(AtA, 2) * 20 + np.linalg.norm(Atb))
print(f"Lipschitz constant L = {L:.4f}")
print(f"Step 1/L = {1/L:.6f}")

gammas_ls = {"γ=0.1": 0.1, "γ=1/β": 1/beta, "γ=1/L": 1/L}
results_2 = {}

for label, gamma in gammas_ls.items():
    history, elapsed, x_final = gradient_descent_ls(x0_ls, A, b, gamma)
    results_2[label] = history
    print(f"\n{label} (γ = {gamma:.6f}):")
    print(f"  Time (s)   : {elapsed:.6f}")
    print(f"  Initial f  : {history[0]:.6f}")
    print(f"  Final f    : {history[-1]:.6f}")

fig2, ax2 = plt.subplots(figsize=(10, 6))
for label, history in results_2.items():
    ax2.semilogy(history, label=label)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("f(x)  [log scale]")
ax2.set_title("Assignment 2 – Least Squares: Objective vs Iteration")
ax2.legend()
ax2.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("assignment2_leastsquares.png", dpi=150)
plt.show()
print("\nPlot saved: assignment2_leastsquares.png")

# ASSIGNMENT 3: Fixed Point Problems
def calculate_objective_fp1(x):
    return 0.5 * (x + np.log(1 + x))**2

def compute_gradient_fp1(x):
    return (x + np.log(1 + x)) * (1 + 1 / (1 + x))

def calculate_objective_fp2(x):
    return 0.5 * (x + np.log(2 + x))**2

def compute_gradient_fp2(x):
    return (x + np.log(2 + x)) * (1 + 1 / (2 + x))

def gradient_descent_fp(x0, calc_obj, calc_grad, gamma, max_iter=100):
    x = float(x0)
    history_f = [calc_obj(x)]
    history_x = [x]
    start = time.time()

    for _ in range(max_iter):
        grad = calc_grad(x)
        x = x - gamma * grad
        history_f.append(calc_obj(x))
        history_x.append(x)

    elapsed = time.time() - start
    return history_f, history_x, elapsed

print("ASSIGNMENT 3: Fixed Point Problems")

x_plot = np.linspace(0, 2, 300)
fig3a, ax3a = plt.subplots(figsize=(8, 5))
ax3a.plot(x_plot, x_plot,           label="y = x",        lw=2)
ax3a.plot(x_plot, np.log(1+x_plot), label="y = ln(1+x)",  lw=2)
ax3a.plot(x_plot, np.log(2+x_plot), label="y = ln(2+x)",  lw=2)
ax3a.set_xlabel("x")
ax3a.set_ylabel("y")
ax3a.set_title("Assignment 3 – Fixed Point: Graph")
ax3a.legend()
ax3a.grid(True, ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("assignment3_fixedpoint_graph.png", dpi=150)
plt.show()

from scipy.optimize import brentq
fp1_exact = 0.0
fp2_exact = brentq(lambda x: x - np.log(2 + x), 0.5, 2)
print(f"\nApproximate fixed point of ln(1+x): x* ≈ {fp1_exact:.6f}  (x=0 is the only solution)")
print(f"Approximate fixed point of ln(2+x): x* ≈ {fp2_exact:.6f}")

x_vals = np.linspace(0.001, 2, 10000)

def g1_second_deriv(x):
    f  = x + np.log(1 + x)
    f1 = 1 + 1 / (1 + x)
    f2 = -1 / (1 + x)**2
    return f1**2 + f * f2

def g2_second_deriv(x):
    f  = x + np.log(2 + x)
    f1 = 1 + 1 / (2 + x)
    f2 = -1 / (2 + x)**2
    return f1**2 + f * f2

L1 = np.max(np.abs(g1_second_deriv(x_vals)))
L2 = np.max(np.abs(g2_second_deriv(x_vals)))
print(f"\nLipschitz constant L1 (for g1) = {L1:.6f}")
print(f"Lipschitz constant L2 (for g2) = {L2:.6f}")
print(f"Step γ1 = 1/L1 = {1/L1:.6f}")
print(f"Step γ2 = 1/L2 = {1/L2:.6f}")

x0_fp = 1.0

history_f1, history_x1, time1 = gradient_descent_fp(
    x0_fp, calculate_objective_fp1, compute_gradient_fp1, 1/L1)

print(f"\n--- g1 with γ = 1/L1 ---")
print(f"{'Iter':>6}  {'f(x)':>14}  {'x':>14}")
for i in range(0, 100, 10):
    print(f"{i:>6}  {history_f1[i]:>14.8f}  {history_x1[i]:>14.8f}")
print(f"Time for 100 iterations: {time1:.6f} s")
print(f"g1 converges to 0: {history_f1[-1] < 1e-6}")

history_f2, history_x2, time2 = gradient_descent_fp(
    x0_fp, calculate_objective_fp2, compute_gradient_fp2, 1/L2)

print(f"\n--- g2 with γ = 1/L2 ---")
print(f"{'Iter':>6}  {'f(x)':>14}  {'x':>14}")
for i in range(0, 100, 10):
    print(f"{i:>6}  {history_f2[i]:>14.8f}  {history_x2[i]:>14.8f}")
print(f"Time for 100 iterations: {time2:.6f} s")
print(f"g2 converges to 0: {history_f2[-1] < 1e-6}")

fig3b, ax3b = plt.subplots(figsize=(10, 6))
ax3b.semilogy(history_f1, label=f"g1  (L1={L1:.2f}, γ=1/L1={1/L1:.4f})")
ax3b.semilogy(history_f2, label=f"g2  (L2={L2:.2f}, γ=1/L2={1/L2:.4f})")
ax3b.set_xlabel("Iteration")
ax3b.set_ylabel("g(x)  [log scale]")
ax3b.set_title("Assignment 3 – Fixed Point: g1 and g2 vs Iteration")
ax3b.legend()
ax3b.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("assignment3_fixedpoint_convergence.png", dpi=150)
plt.show()
print("\nPlots saved: assignment3_fixedpoint_graph.png, assignment3_fixedpoint_convergence.png")

print("\n✓ All assignments complete.")