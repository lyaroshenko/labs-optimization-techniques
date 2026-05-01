"""
LAB 3 — Stochastic Gradient Descent & Subgradient Descent
Optimization Techniques, Spring 2025/2026
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
np.random.seed(42)
# We want to find a straight line:   weight = w0 + w1 * height
# w0 = intercept (where the line crosses the y-axis)
# w1 = slope     (how much weight changes per inch of height)
#
# We find w0 and w1 by MINIMIZING an error function.
# We try two error functions:
#   MSE (Mean Squared Error)   — squares the errors, sensitive to outliers
#   MAE (Mean Absolute Error)  — uses absolute values, more robust
#
# four algorithms to minimize these:
#   Task 1: Gradient Descent          → minimizes MSE
#   Task 2: Stochastic Gradient Descent → minimizes MSE (one sample at a time)
#   Task 3: Subgradient Descent       → minimizes MAE
#   Task 4: Stochastic Subgradient    → minimizes MAE (one sample at a time)

# STEP 1 — LOAD OR CREATE DATA
def load_data():
    """
    load the Kaggle CSV file.
    Returns:
        heights : array of height values (in inches)
        weights : array of weight values (in lbs)
    """
    if os.path.exists("weight-height.csv"):
        df = pd.read_csv("weight-height.csv")
        df.columns = [col.strip().lower() for col in df.columns]
        heights = df["height"].values.astype(float)
        weights = df["weight"].values.astype(float)
        print(f"Loaded data")
    return heights, weights

# Load the data
heights, weights = load_data()
n = len(heights)

print(f"Number of samples: {n}")
print(f"Average height: {heights.mean():.2f} inches")
print(f"Average weight: {weights.mean():.2f} lbs")

# STEP 2 — BUILD THE DESIGN MATRIX
# Instead of working with w0 and w1 separately, we pack them into one vector:
#   omega = [w0, w1]
#
# And we build a matrix X where:
#   - First column is all 1s (for w0, the intercept)
#   - Second column is the heights (for w1, the slope)
#
# Then the prediction for ALL samples at once is just:  X @ omega
# (matrix multiplication)
#
# Example with 3 people:
#   X = [[1, 60],        omega = [w0]     prediction = [w0 + w1*60]
#        [1, 65],                [w1]                   [w0 + w1*65]
#        [1, 70]]                                        [w0 + w1*70]

# Build the X matrix
ones_column = np.ones(n)                      # column of 1s (length n)
X = np.column_stack([ones_column, heights])   # shape: (n, 2)
y = weights.copy()                            # our targets (weight values)

print(f"\nX matrix shape: {X.shape}  (rows=samples, cols=[1, height])")
print(f"y vector shape: {y.shape}  (one weight per sample)\n")


# TASK 1 — GRADIENT DESCENT FOR MSE
#
# MSE (Mean Squared Error):
#   S(omega) = 1/(2n) * sum of (actual - predicted)^2
#            = 1/(2n) * ||y - X*omega||^2
#
# To minimize S, we compute its gradient and step in the opposite direction:
#   gradient = -(1/n) * X^T * (y - X*omega)
#   omega_new = omega_old - step_size * gradient
#
# Step size: gamma = 1/beta
#   where beta = largest eigenvalue of (X^T * X / n)

def calculate_mse(omega, X, y):
    """
    Compute the MSE cost: S(omega) = 1/(2n) * ||y - X*omega||^2

    omega : [w0, w1] — our current guess for the parameters
    X     : design matrix (n x 2)
    y     : target weights (length n)
    Returns a single number (the error)
    """
    n = len(y)

    # Compute residuals (errors): how far off each prediction is
    errors = y - X @ omega     # X @ omega = predicted weights for all n people

    # MSE = average of squared errors (with 1/2 factor for clean gradient formula)
    mse = (1.0 / (2 * n)) * np.sum(errors ** 2)

    return mse


def compute_gradient_mse(omega, X, y):
    """
    Compute the gradient of MSE with respect to omega.
    gradient = -(1/n) * X^T * errors

    Returns a vector of length 2: [d/dw0, d/dw1]
    """
    n = len(y)

    errors = y - X @ omega

    # X.T is (2 x n), errors is (n,), so X.T @ errors is (2,)
    gradient = -(1.0 / n) * (X.T @ errors)

    return gradient


def gradient_descent(omega_start, X, y, step_size, num_iterations):
    """
    Run full gradient descent for MSE.

    omega_start    : starting guess for [w0, w1]
    step_size      : how big each step is (gamma)
    num_iterations : how many steps to take

    Returns:
        omega   : the final [w0, w1] found
        history : list of MSE values at each step (for plotting convergence)
    """
    omega = omega_start.copy()

    # Record the starting error
    history = [calculate_mse(omega, X, y)]

    for step in range(num_iterations):
        # Compute gradient at current omega
        grad = compute_gradient_mse(omega, X, y)

        # Move in the opposite direction of the gradient
        omega = omega - step_size * grad

        # Record the new error
        history.append(calculate_mse(omega, X, y))

    return omega, history


# --- Compute the safe step size ---
XTX_over_n = X.T @ X / n
eigenvalues = np.linalg.eigvalsh(XTX_over_n)   # compute all eigenvalues
beta = float(eigenvalues.max())                  # take the largest one
step_size_gd = 1.0 / beta                        # safe step size

print("=" * 55)
print("TASK 1 — Gradient Descent (MSE)")
print(f"  Lipschitz constant beta = {beta:.6f}")
print(f"  Step size (1/beta)      = {step_size_gd:.6f}")

# Starting point: both parameters = 0
omega_start = np.zeros(2)
num_iter_gd = 1000

# Run gradient descent
omega_gd, history_gd = gradient_descent(
    omega_start, X, y, step_size_gd, num_iter_gd
)

print(f"  Found: w0 = {omega_gd[0]:.4f},  w1 = {omega_gd[1]:.4f}")
print(f"  Line: weight = {omega_gd[0]:.2f} + {omega_gd[1]:.4f} * height")
print(f"  Final MSE = {history_gd[-1]:.4f}\n")


# TASK 2 — STOCHASTIC GRADIENT DESCENT (SGD) FOR MSE
#
# Regular GD uses ALL n samples to compute each gradient update.
# SGD uses just ONE randomly chosen sample per update.
#
# This is much faster per step (especially with large datasets),
# but the gradient is noisy — it zigzags more but gets there faster overall.
#
# Per-sample gradient:
#   Pick random sample i
#   error_i = y_i - (w0 + w1 * x_i)
#   gradient_i = -error_i * [1, x_i]
#   omega_new = omega - step_size * gradient_i

def compute_stochastic_gradient_mse(omega, x_i, y_i):
    """
    Compute the gradient using just ONE data point (sample i).
    omega : current [w0, w1]
    x_i   : height of sample i
    y_i   : weight of sample i
    Returns gradient vector of length 2.
    """
    # Feature vector for this one sample: [1, x_i]
    phi_i = np.array([1.0, x_i])

    # Prediction error for this one sample
    error_i = y_i - float(phi_i @ omega)

    # Per-sample gradient
    gradient_i = -error_i * phi_i

    return gradient_i

def stochastic_gradient_descent_mse(omega_start, X, y, step_size, num_iterations):
    """
    Run SGD for MSE.

    Instead of evaluating the full MSE every step (too slow),
    we evaluate it every 2000 steps and record that.

    Returns:
        omega   : final [w0, w1]
        history : list of (step_number, mse_value) tuples
    """
    n = len(y)
    omega = omega_start.copy()

    # Record starting point
    history = [(0, calculate_mse(omega, X, y))]

    evaluate_every = 2000   # check MSE every this many steps

    for step in range(1, num_iterations + 1):
        # Pick a RANDOM sample index
        i = np.random.randint(0, n)

        # Compute gradient using just that one sample
        grad = compute_stochastic_gradient_mse(omega, X[i, 1], y[i])

        # Update omega
        omega = omega - step_size * grad

        # Every 2000 steps, record the current MSE
        if step % evaluate_every == 0:
            current_mse = calculate_mse(omega, X, y)
            history.append((step, current_mse))

    return omega, history


print("TASK 2 — Stochastic Gradient Descent (MSE)")

# SGD works with a smaller step size to stay stable despite noisy gradients
step_size_sgd = step_size_gd * 0.1
num_iter_sgd  = 200_000

omega_sgd, history_sgd = stochastic_gradient_descent_mse(
    omega_start, X, y, step_size_sgd, num_iter_sgd
)

print(f"  Step size = {step_size_sgd:.6f}")
print(f"  Found: w0 = {omega_sgd[0]:.4f},  w1 = {omega_sgd[1]:.4f}")
print(f"  Final MSE = {calculate_mse(omega_sgd, X, y):.4f}\n")

# TASK 3 — SUBGRADIENT DESCENT FOR MAE
#
# MAE (Mean Absolute Error):
#   A(omega) = 1/n * sum of |actual - predicted|
#            = 1/n * sum of |errors|
#
# Problem: MAE is NOT differentiable at points where error = 0.
#   (the |x| function has a sharp corner at x=0 — no single slope there)
#
# Solution: use a SUBGRADIENT instead of a gradient.
#   A subgradient is any value that "fits under" the function at that point.
#   For |x|:  subgradient = +1 if x > 0
#                          = -1 if x < 0
#                          =  0 if x = 0
# Subgradient of MAE:
#   g = -(1/n) * X^T * sign(errors)
#
# Step size: MUST shrink over time for subgradient to converge.
#   We use: gamma_k = gamma_0 / sqrt(k+1)
#   (gets smaller each step but never reaches zero — this guarantees convergence)

def calculate_mae(omega, X, y):
    """
    Compute MAE: A(omega) = 1/n * sum(|y - X*omega|)
    """
    n = len(y)
    errors = y - X @ omega
    mae = np.sum(np.abs(errors)) / n
    return float(mae)

def compute_subgradient_mae(omega, X, y):
    """
    Compute the subgradient of MAE.
    g = -(1/n) * X^T * sign(errors)

    np.sign(x) returns +1, -1, or 0 — valid subgradient for |x|.
    """
    n = len(y)
    errors = y - X @ omega

    # sign of each error: +1 (under-predicted), -1 (over-predicted), 0 (exact)
    signs = np.sign(errors)

    subgradient = -(1.0 / n) * (X.T @ signs)
    return subgradient


def subgradient_descent_mae(omega_start, X, y, gamma_0, num_iterations):
    """
    Run subgradient descent for MAE.

    gamma_0        : starting step size (will shrink each iteration)
    num_iterations : number of steps

    Returns:
        omega_best : the best [w0, w1] found at any point during the run
        history    : list of best MAE values seen so far at each step
    """
    omega = omega_start.copy()

    # Track the best solution found so far
    best_mae   = calculate_mae(omega, X, y)
    omega_best = omega.copy()

    history = [best_mae]

    # Count how many times we hit a non-differentiable point (error == 0)
    non_diff_count = 0

    errors = y - X @ omega   # compute once at start, update each step

    for k in range(num_iterations):

        # Check if any residual is exactly 0 (non-differentiable point)
        if np.any(errors == 0.0):
            non_diff_count += 1

        # Shrinking step size: gets smaller as k increases
        gamma_k = gamma_0 / np.sqrt(k + 1)

        # Compute subgradient
        signs = np.sign(errors)
        subgrad = -(1.0 / len(y)) * (X.T @ signs)

        # Update omega
        omega = omega - gamma_k * subgrad

        # Recompute errors after the update
        errors = y - X @ omega

        # Check if this is better than our best so far
        current_mae = float(np.sum(np.abs(errors))) / len(y)
        if current_mae < best_mae:
            best_mae   = current_mae
            omega_best = omega.copy()

        history.append(best_mae)

    print(f"  Non-differentiable points hit: {non_diff_count}")
    return omega_best, history


print("TASK 3 — Subgradient Descent (MAE)")

# Start with a larger gamma_0 — it shrinks automatically
gamma_0_sub  = step_size_gd * 50
num_iter_sub = 5000

omega_mae, history_mae = subgradient_descent_mae(
    omega_start, X, y, gamma_0_sub, num_iter_sub
)

print(f"  Starting step size gamma_0 = {gamma_0_sub:.4f}")
print(f"  (shrinks each step as gamma_0 / sqrt(k+1))")
print(f"  Found: w0 = {omega_mae[0]:.4f},  w1 = {omega_mae[1]:.4f}")
print(f"  Final MAE = {history_mae[-1]:.4f}")

# Answer question (a): is MAE fit better?
mae_of_gd_solution  = calculate_mae(omega_gd, X, y)
mae_of_mae_solution = history_mae[-1]
print(f"\n  (a) Is MAE fit better than MSE fit?")
print(f"      MAE when using GD-MSE solution  : {mae_of_gd_solution:.4f}")
print(f"      MAE when using Subgrad solution : {mae_of_mae_solution:.4f}")
print(f"      → Yes, the MAE solution has lower MAE (it directly optimises it)")
print(f"         Both give similar lines on clean data, but MAE is more")
print(f"         robust when outliers exist (big errors don't get amplified)\n")


# TASK 4 — STOCHASTIC SUBGRADIENT DESCENT FOR MAE
#
# Same idea as Task 3, but use ONE random sample at a time (like SGD in Task 2).
#
# Per-sample subgradient of |y_i - w0 - w1*x_i|:
#   g_i = -sign(y_i - predicted_i) * [1, x_i]
#
# This is even noisier than Task 3 (batch), but scales better to huge datasets.


def compute_stochastic_subgradient_mae(omega, x_i, y_i):
    """
    Compute the subgradient using just ONE data point.

    omega : current [w0, w1]
    x_i   : height of sample i
    y_i   : weight of sample i
    """
    phi_i   = np.array([1.0, x_i])                # feature vector [1, x_i]
    error_i = y_i - float(phi_i @ omega)           # prediction error
    return -np.sign(error_i) * phi_i               # per-sample subgradient


def stochastic_subgradient_descent_mae(omega_start, X, y, gamma_0, num_iterations):
    """
    Run stochastic subgradient descent for MAE.
    Uses one random sample per step.
    Step size shrinks: gamma_k = gamma_0 / sqrt(k).

    Returns:
        omega_best : best [w0, w1] found
        history    : list of (step_number, best_mae) tuples
    """
    n_data = len(y)
    omega  = omega_start.copy()

    best_mae   = calculate_mae(omega, X, y)
    omega_best = omega.copy()

    history       = [(0, best_mae)]
    non_diff_count = 0
    evaluate_every = 2000

    for k in range(1, num_iterations + 1):
        # Pick a random sample
        i = np.random.randint(0, n_data)

        # Check for non-differentiable point
        error_i = y[i] - float(X[i] @ omega)
        if error_i == 0.0:
            non_diff_count += 1

        # Shrinking step size
        gamma_k = gamma_0 / np.sqrt(k)

        # Per-sample subgradient
        subgrad = compute_stochastic_subgradient_mae(omega, X[i, 1], y[i])

        # Update
        omega = omega - gamma_k * subgrad

        # Periodically check if we found a new best
        if k % evaluate_every == 0:
            current_mae = calculate_mae(omega, X, y)
            if current_mae < best_mae:
                best_mae   = current_mae
                omega_best = omega.copy()
            history.append((k, best_mae))

    print(f"  Non-differentiable points hit: {non_diff_count}")
    return omega_best, history

print("TASK 4 — Stochastic Subgradient Descent (MAE)")

gamma_0_ssub  = gamma_0_sub
num_iter_ssub = 200_000

omega_smae, history_smae = stochastic_subgradient_descent_mae(
    omega_start, X, y, gamma_0_ssub, num_iter_ssub
)

print(f"  Found: w0 = {omega_smae[0]:.4f},  w1 = {omega_smae[1]:.4f}")
print(f"  Final MAE = {calculate_mae(omega_smae, X, y):.4f}\n")

# Compare batch vs stochastic on MAE
print("  Comparison — Batch Subgradient vs Stochastic Subgradient (MAE):")
print(f"    Batch  [w0={omega_mae[0]:.4f}, w1={omega_mae[1]:.4f}]"
      f"  MAE={calculate_mae(omega_mae, X, y):.4f}")
print(f"    Stoch  [w0={omega_smae[0]:.4f}, w1={omega_smae[1]:.4f}]"
      f"  MAE={calculate_mae(omega_smae, X, y):.4f}")


# RESULTS SUMMARY TABLE
print("RESULTS SUMMARY")
print(f"{'Method':<28} {'w0':>10} {'w1':>10} {'MSE':>10} {'MAE':>10}")

all_solutions = [
    ("GD (MSE)",            omega_gd),
    ("SGD (MSE)",           omega_sgd),
    ("Subgradient (MAE)",   omega_mae),
    ("Stoch Subgrad (MAE)", omega_smae),
]

for name, omega in all_solutions:
    mse = calculate_mse(omega, X, y)
    mae = calculate_mae(omega, X, y)
    print(f"{name:<28} {omega[0]:>10.4f} {omega[1]:>10.4f} {mse:>10.4f} {mae:>10.4f}")
# PLOTTING — 4 panels
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Lab 3 — SGD & Subgradient Descent on Height–Weight Data",
             fontsize=14, fontweight="bold")
C_GD   = "#2563eb"   # blue
C_SGD  = "#16a34a"   # green
C_MAE  = "#dc2626"   # red
C_SMAE = "#9333ea"   # purple

# Build x values for plotting the regression lines
x_plot = np.linspace(heights.min() - 1, heights.max() + 1, 300)

# ── Panel 1 (top-left): Data + all 4 fitted lines ──────────────────────────
ax = axes[0, 0]
ax.scatter(heights, weights, color="lightgray", s=3, alpha=0.4,
           label="Data", zorder=1)

ax.plot(x_plot, omega_gd[0]   + omega_gd[1]   * x_plot,
        color=C_GD,   lw=2,    label=f"GD MSE  (w1={omega_gd[1]:.3f})")
ax.plot(x_plot, omega_sgd[0]  + omega_sgd[1]  * x_plot,
        color=C_SGD,  lw=2, ls="--", label=f"SGD MSE (w1={omega_sgd[1]:.3f})")
ax.plot(x_plot, omega_mae[0]  + omega_mae[1]  * x_plot,
        color=C_MAE,  lw=2, ls="-.", label=f"Subgrad MAE (w1={omega_mae[1]:.3f})")
ax.plot(x_plot, omega_smae[0] + omega_smae[1] * x_plot,
        color=C_SMAE, lw=2, ls=":",  label=f"Stoch Subgrad (w1={omega_smae[1]:.3f})")

ax.set_xlabel("Height (inches)")
ax.set_ylabel("Weight (lbs)")
ax.set_title("All 4 Fitted Lines on Data")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Panel 2 (top-right): GD convergence curve 
ax = axes[0, 1]
ax.plot(history_gd, color=C_GD, lw=2)
ax.set_xlabel("Iteration")
ax.set_ylabel("MSE")
ax.set_title("Task 1: GD Convergence (MSE)")
ax.grid(True, alpha=0.3)

# ── Panel 3 (bottom-left): SGD convergence curve
ax = axes[1, 0]
steps_sgd = [h[0] for h in history_sgd]   # x-axis: step numbers
vals_sgd  = [h[1] for h in history_sgd]   # y-axis: MSE values
ax.plot(steps_sgd, vals_sgd, color=C_SGD, lw=2)
ax.set_xlabel("Iteration")
ax.set_ylabel("MSE")
ax.set_title("Task 2: SGD Convergence (MSE)")
ax.grid(True, alpha=0.3)

# ── Panel 4 (bottom-right): Subgradient convergence (both batch & stochastic)
ax = axes[1, 1]

# Batch subgradient: one MAE value per iteration
ax.plot(history_mae, color=C_MAE, lw=2, label="Batch Subgrad MAE")

# Stochastic: stored as (step, mae) pairs
steps_smae = [h[0] for h in history_smae]
vals_smae  = [h[1] for h in history_smae]
ax.plot(steps_smae, vals_smae, color=C_SMAE, lw=2, ls="--",
        label="Stoch Subgrad MAE")

ax.set_xlabel("Iteration")
ax.set_ylabel("Best MAE so far")
ax.set_title("Tasks 3 & 4: Subgradient Convergence (MAE)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()

out_path = "lab3_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved as '{out_path}'")