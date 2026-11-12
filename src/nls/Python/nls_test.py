from galahad import nls
import numpy as np
import pytest


# test problem:
# minimize 1/2 sum_i w_i c_i(x)^2, where
# c(x) = (x_0^2 + p, x_0 + x_1^2, x_0 - x_1)
p = 1.0
n = 2
m = 3
w = np.array([1.0, 1.0, 1.0])

# Jacobian sparsity (coordinate format)
J_type = 'coordinate'
J_ne = 5
J_row = np.array([0, 1, 1, 2, 2])
J_col = np.array([0, 0, 1, 0, 1])
J_ptr = None

# sparsity of the Hessian sum_i y_i H_i(x) of the residuals
# (lower triangle, coordinate format)
H_type = 'coordinate'
H_ne = 2
H_row = np.array([0, 1])
H_col = np.array([0, 1])
H_ptr = None

# sparsity of the residual-Hessians-vector product matrix
# (H_0(x) v, ..., H_{m-1}(x) v), stored by columns
P_type = 'sparse_by_columns'
P_ne = 2
P_row = np.array([0, 1])
P_col = None
P_ptr = np.array([0, 1, 2, 2])

def eval_c(x):
    return np.array([x[0]**2 + p, x[0] + x[1]**2, x[0] - x[1]])

def eval_j(x):
    return np.array([2. * x[0], 1., 2. * x[1], 1., -1.])

def eval_h(x, y):
    return np.array([2. * y[0], 2. * y[1]])

def eval_hprod(x, v):
    return np.array([2. * v[0], 2. * v[1]])

def jacobian(x):
    J = np.zeros((m, n))
    J[J_row, J_col] = eval_j(x)
    return J

# nls is a local method, so it is only required to find a local minimizer.
# Here the minimizer is unique: since (x_0^2 + p)^2 >= p^2, the objective
# is at least p^2 / 2, with equality only when x_0 = 0 and then x_1 = 0
# (from the other two residuals).
x_star = np.array([0.0, 0.0])
c_star = np.array([p, 0.0, 0.0])
f_star = 0.5 * p**2

@pytest.fixture
def solution():
    options = nls.initialize()
    options['jacobian_available'] = 2
    options['hessian_available'] = 2
    options['model'] = 6
    nls.load(n, m,
             J_type, J_ne, J_row, J_col, J_ptr,
             H_type, H_ne, H_row, H_col, H_ptr,
             P_type, P_ne, P_row, P_col, P_ptr,
             w, options)
    x0 = np.array([1.5, 1.5])
    x, c, g = nls.solve(n, m, x0, eval_c, J_ne, eval_j, H_ne, eval_h,
                        P_ne, eval_hprod)
    inform = nls.information()
    nls.terminate()
    return x, c, g, inform


def test_nls_status(solution):
    x, c, g, inform = solution
    assert inform['status'] == 0


def test_nls_solution(solution):
    x, c, g, inform = solution
    # returned values are consistent with the residuals at x
    assert c == pytest.approx(eval_c(x), abs=1e-8)
    J = jacobian(x)
    assert g == pytest.approx(J.T @ (w * c), abs=1e-8)
    assert inform['obj'] == pytest.approx(0.5 * np.sum(w * eval_c(x)**2))
    # first-order optimality
    assert g == pytest.approx(np.zeros(n), abs=1e-5)
    # second-order: the Hessian J^T W J + sum_i w_i c_i H_i of the
    # objective is positive definite
    H = np.zeros((n, n))
    H[H_row, H_col] = eval_h(x, w * c)
    H = H + np.tril(H, -1).T
    assert np.all(np.linalg.eigvalsh(J.T @ (w[:, None] * J) + H) > 0.0)
    assert x == pytest.approx(x_star, abs=1e-5)
    assert c == pytest.approx(c_star, abs=1e-5)
    assert inform['obj'] == pytest.approx(f_star, abs=1e-8)
