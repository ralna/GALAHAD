from galahad import blls
import numpy as np
import pytest


# test problem:
# minimize 1/2 || A_o x - b ||_W^2 + sigma/2 || x - x_s ||_2^2
# subject to x_l <= x <= x_u, where
# A_o = (  I  ) and b = ( 1, 2, ..., n ),
#       ( e^T )         (     n + 1    )
# W = diag(2, 1, ..., 1), x_s = 0, -1 <= x_0 <= 1, x_1 is free,
# and x_j <= 2 for j >= 2
n = 10
o = n + 1
sigma = 1.0
infinity = float("inf")
A = np.vstack([np.eye(n), np.ones((1, n))])
b = np.r_[np.arange(1, n + 1), n + 1].astype(float)
w = np.r_[2.0, np.ones(o - 1)]
x_s = np.zeros(n)
x_l = np.r_[-1.0, np.full(n - 1, -infinity)]
x_u = np.r_[1.0, infinity, np.full(n - 2, 2.0)]

def storage(Ao_type):
    # return Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr, Ao_val for A
    # in the given storage scheme
    if Ao_type in ('dense', 'dense_by_rows'):
        return o * n, None, None, 0, None, A.ravel()
    if Ao_type == 'dense_by_columns':
        return o * n, None, None, 0, None, A.ravel(order='F')
    if Ao_type == 'sparse_by_columns':
        cols, rows = np.nonzero(A.T)
    else:
        rows, cols = np.nonzero(A)
    vals = A[rows, cols]
    if Ao_type == 'coordinate':
        return len(vals), rows, cols, 0, None, vals
    if Ao_type == 'sparse_by_rows':
        ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=o))]
        return len(vals), None, cols, o + 1, ptr, vals
    ptr = np.r_[0, np.cumsum(np.bincount(cols, minlength=n))]
    return len(vals), rows, None, n + 1, ptr, vals

def gradient(x):
    return A.T @ (w * (A @ x - b)) + sigma * (x - x_s)

def objective(x):
    r = A @ x - b
    return 0.5 * np.sum(w * r**2) + 0.5 * sigma * np.sum((x - x_s)**2)

def solve(Ao_type):
    Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr, Ao_val = storage(Ao_type)
    options = blls.initialize()
    # use the LAPACK solvers sytr and potr rather than the defaults, which
    # need the environment variables OMP_CANCELLATION and OMP_PROC_BIND set.
    options['sbls_options']['symmetric_linear_solver'] = 'sytr'
    options['sbls_options']['definite_linear_solver'] = 'potr'
    blls.load(n, o, Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr,
              options)
    x0 = np.zeros(n)
    z0 = np.zeros(n)
    x, z, r, g, x_stat = blls.solve(n, o, Ao_ne, Ao_val, b, sigma,
                                    x_l, x_u, x0, z0, w, x_s)
    inform = blls.information()
    blls.terminate()
    return x, z, r, g, x_stat, inform

# blls is a local method, so it is only required to find a local minimizer,
# but here the objective is strictly convex (its Hessian A_o^T W A_o + sigma I
# is positive definite), so the minimizer is unique. With t = e^T x - (n + 1),
# g_0 = 3 x_0 - 2 + t and g_j = 2 x_j - (j + 1) + t for j >= 1, so the free
# components satisfy x_0 = (2 - t) / 3 and x_j = (j + 1 - t) / 2. With
# x_6, ..., x_9 on their upper bounds, this gives t = 2, and so x_5 = 2 is
# also on its upper bound, but with g_5 = 0 (a degenerate bound), while
# z_j = g_j = 5 - j < 0 for j >= 6 as required at upper bounds
x_star = np.r_[0.0, 0.0, 0.5, 1.0, 1.5, np.full(n - 5, 2.0)]
z_star = np.r_[np.zeros(6), 5.0 - np.arange(6, n)]
r_star = A @ x_star - b
f_star = objective(x_star)

STORAGE = ['coordinate', 'sparse_by_rows', 'sparse_by_columns',
           'dense_by_rows', 'dense_by_columns', 'dense']

@pytest.fixture(params=STORAGE)
def solution(request):
    return solve(request.param)


def test_blls_status(solution):
    x, z, r, g, x_stat, inform = solution
    assert inform['status'] == 0


def test_blls_solution(solution):
    x, z, r, g, x_stat, inform = solution
    # returned values are consistent with the objective at x
    assert r == pytest.approx(A @ x - b, abs=1e-8)
    assert g == pytest.approx(gradient(x), abs=1e-8)
    assert inform['obj'] == pytest.approx(objective(x))
    # first-order optimality: x is feasible, g = z, and z_j is zero, at
    # least zero or at most zero when x_j is free, on its lower bound or on
    # its upper bound respectively; x_stat > 0 flags the strongly active
    # upper bounds x_6, ..., x_9
    assert np.all(x >= x_l) and np.all(x <= x_u)
    assert g == pytest.approx(z, abs=1e-8)
    free = (x > x_l + 1e-8) & (x < x_u - 1e-8)
    assert z[free] == pytest.approx(np.zeros(np.sum(free)), abs=1e-6)
    assert np.all(z[x >= x_u - 1e-8] <= 1e-6)
    assert np.all(z[x <= x_l + 1e-8] >= -1e-6)
    assert np.all(x_stat[:5] == 0) and np.all(x_stat[6:] > 0)
    assert x == pytest.approx(x_star, abs=1e-6)
    assert z == pytest.approx(z_star, abs=1e-6)
    assert r == pytest.approx(r_star, abs=1e-6)
    assert inform['obj'] == pytest.approx(f_star, abs=1e-8)
