"""
HW3, Member 2: stationary iterative solvers for A x = b, written
from scratch (Jacobi, Gauss-Seidel) with the stopping criteria of the course.

Functions only, no top-level execution.

Interface (shared with Member 3):
    x, history = jacobi(A, b, x0=None, tol=1e-8, max_iter=1000,
                        criterion="residual", mode="matrix", norm=np.inf)
    x, history = gauss_seidel(A, b, ...same arguments...)
history is a pandas DataFrame with one row per iteration and columns
iter, rel_change, abs_change, rel_residual, time_s. All three criteria are
logged at every iteration. history.attrs["converged"] (bool) and
history.attrs["reason"] (str) tell why the loop stopped.
Optional x_ref (a reference solution) adds the column
rel_error = ||x - x_ref|| / ||x_ref|| (true error, for validation only).

Example (from the repository root):
    from hw3_system import build_system, initial_opinions
    from solvers import jacobi
    alpha = 0.85
    A, W = build_system(alpha)
    b = (1 - alpha) * initial_opinions(A.shape[0])
    x, history = jacobi(A, b, tol=1e-8)
    print(history.attrs["converged"], history.attrs["reason"])

Jacobi needs about log(tol) / log(alpha) iterations on this system, so raise
max_iter for alpha close to 1 (e.g. about 1833 for alpha = 0.99, tol = 1e-8).
"""

import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular

CRITERIA = ("residual", "rel_change", "abs_change")


def _iterate(update, A, b, x0, tol, max_iter, criterion, norm, x_ref=None):
    """
    Generic stationary iteration x^(k+1) = update(x^(k)) with logging and
    stopping criteria (Festa's definitions):
      rel_change   = ||x^(k+1) - x^(k)|| / ||x^(k+1)||
      abs_change   = ||x^(k+1) - x^(k)||
      rel_residual = ||b - A x^(k+1)|| / ||b||
    The loop stops when the chosen criterion is <= tol, or after max_iter.

    :param update: function x_old -> x_new (one Jacobi or Gauss-Seidel sweep)
    :param A: scipy CSR matrix, only used for the residual
    :param b: 1D numpy array
    :param x0: initial guess (1D numpy array) or None for zeros
    :param tol: tolerance
    :param max_iter: maximum number of iterations
    :param criterion: "residual" | "rel_change" | "abs_change"
    :param norm: order of the vector norm (np.inf, 2, 1)
    :param x_ref: optional reference solution, logs the true relative error
    :return x: last iterate
    :return history: pandas DataFrame, see module docstring
    """
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}")

    x = np.zeros_like(b, dtype=np.float64) if x0 is None else np.array(x0, dtype=np.float64)
    norm_b = np.linalg.norm(b, norm)
    column = {"residual": "rel_residual", "rel_change": "rel_change", "abs_change": "abs_change"}[criterion]

    rows = []
    converged = False
    for k in range(1, max_iter + 1):
        start = time.perf_counter()
        x_new = update(x)
        iter_time = time.perf_counter() - start  # time of the sweep only, not of the logging

        abs_change = np.linalg.norm(x_new - x, norm)
        row = {"iter": k,
               "rel_change": abs_change / np.linalg.norm(x_new, norm),
               "abs_change": abs_change,
               "rel_residual": np.linalg.norm(b - A @ x_new, norm) / norm_b,
               "time_s": iter_time}
        if x_ref is not None:
            row["rel_error"] = np.linalg.norm(x_new - x_ref, norm) / np.linalg.norm(x_ref, norm)
        rows.append(row)
        x = x_new

        if row[column] <= tol:
            converged = True
            break

    history = pd.DataFrame(rows)
    history.attrs["converged"] = converged
    history.attrs["reason"] = (f"{criterion} <= {tol:g} at iteration {k}" if converged
                               else f"max_iter = {max_iter} reached ({column} = {row[column]:.2e})")
    return x, history


def _jacobi_sweep(indptr, indices, data, diag, b, x, x_new):
    """
    One Jacobi sweep, element by element on the CSR arrays of A:
        x_new_i = (b_i - sum_{j != i} a_ij x_j) / a_ii
    Only the old vector x is read, so the order of the rows does not matter.
    Plain Python on numpy arrays, so it can also be compiled with numba.njit.

    :param indptr, indices, data: CSR arrays of A
    :param diag: diagonal of A
    :param b: right-hand side
    :param x: current iterate (read only)
    :param x_new: output array, overwritten
    :return x_new: the new iterate
    """
    n = len(b)
    for i in range(n):
        sigma = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            if j != i:
                sigma += data[k] * x[j]
        x_new[i] = (b[i] - sigma) / diag[i]
    return x_new


def jacobi(A, b, x0=None, tol=1e-8, max_iter=1000, criterion="residual", mode="matrix", norm=np.inf, x_ref=None):
    """
    Jacobi method. With the splitting A = D - (D - A), D = diag(A):
        x^(k+1) = D^{-1} (b - (A - D) x^(k))
    i.e. iteration matrix B_J = D^{-1}(D - A). For A = I - alpha W, D = I and
    B_J = alpha W. Converges for any x0 if A is strictly diagonally dominant.

    :param A: scipy sparse matrix (converted to CSR), nonzero diagonal
    :param b: 1D numpy array
    :param x0: initial guess, default zeros
    :param tol: tolerance of the stopping criterion
    :param max_iter: maximum number of iterations
    :param criterion: "residual" | "rel_change" | "abs_change"
    :param mode: "matrix" (sparse mat-vec) | "loop" (element by element, slow)
    :param norm: order of the vector norm used by the criteria
    :param x_ref: optional reference solution, adds the column rel_error to history
    :return x: approximate solution
    :return history: pandas DataFrame (iter, rel_change, abs_change,
        rel_residual, time_s), with attrs "converged" and "reason"
    """
    A = sp.csr_matrix(A)
    diag = A.diagonal()
    if np.any(diag == 0):
        raise ValueError("Jacobi needs a nonzero diagonal")

    if mode == "matrix":
        R = (A - sp.diags(diag)).tocsr()  # off-diagonal part A - D

        def update(x):
            return (b - R @ x) / diag

    elif mode == "loop":
        def update(x):
            return _jacobi_sweep(A.indptr, A.indices, A.data, diag, b, x, np.empty_like(x))

    else:
        raise ValueError('mode must be "matrix" or "loop"')

    return _iterate(update, A, b, x0, tol, max_iter, criterion, norm, x_ref)


def _gauss_seidel_sweep(indptr, indices, data, diag, b, x):
    """
    One Gauss-Seidel sweep, element by element on the CSR arrays of A, IN PLACE:
        x_i = (b_i - sum_{j < i} a_ij x_j^new - sum_{j > i} a_ij x_j^old) / a_ii
    Since x is overwritten row by row, when row i is processed x_j already
    holds the new value for j < i and the old value for j > i.
    Plain Python on numpy arrays, so it can also be compiled with numba.njit.

    :param indptr, indices, data: CSR arrays of A
    :param diag: diagonal of A
    :param b: right-hand side
    :param x: current iterate, overwritten with the new one
    :return x: the new iterate
    """
    n = len(b)
    for i in range(n):
        sigma = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            if j != i:
                sigma += data[k] * x[j]
        x[i] = (b[i] - sigma) / diag[i]
    return x


def gauss_seidel(A, b, x0=None, tol=1e-8, max_iter=1000, criterion="residual", mode="matrix", norm=np.inf, x_ref=None):
    """
    Gauss-Seidel method. With A = L + D + U (strictly lower, diagonal,
    strictly upper part):
        (D + L) x^(k+1) = b - U x^(k)
    i.e. iteration matrix B_GS = -(D + L)^{-1} U. For A = I - alpha W this is
    (I - alpha W_L) x^(k+1) = b + alpha W_U x^(k). Converges for any x0 if A is
    strictly diagonally dominant. Unlike Jacobi, the result of each sweep
    depends on the ordering of the unknowns.

    :param A: scipy sparse matrix (converted to CSR), nonzero diagonal
    :param b: 1D numpy array
    :param x0: initial guess, default zeros
    :param tol: tolerance of the stopping criterion
    :param max_iter: maximum number of iterations
    :param criterion: "residual" | "rel_change" | "abs_change"
    :param mode: "matrix" (sparse triangular solve) | "loop" (element by element, slow)
    :param norm: order of the vector norm used by the criteria
    :param x_ref: optional reference solution, adds the column rel_error to history
    :return x: approximate solution
    :return history: pandas DataFrame (iter, rel_change, abs_change,
        rel_residual, time_s), with attrs "converged" and "reason"
    """
    A = sp.csr_matrix(A)
    diag = A.diagonal()
    if np.any(diag == 0):
        raise ValueError("Gauss-Seidel needs a nonzero diagonal")

    if mode == "matrix":
        DL = sp.tril(A, k=0, format="csr")  # D + L
        U = sp.triu(A, k=1, format="csr")   # strictly upper part

        def update(x):
            return spsolve_triangular(DL, b - U @ x, lower=True)

    elif mode == "loop":
        def update(x):
            return _gauss_seidel_sweep(A.indptr, A.indices, A.data, diag, b, x.copy())

    else:
        raise ValueError('mode must be "matrix" or "loop"')

    return _iterate(update, A, b, x0, tol, max_iter, criterion, norm, x_ref)
