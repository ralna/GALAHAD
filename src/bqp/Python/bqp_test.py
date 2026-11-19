from galahad import bqp
import numpy as np
import pytest


# test problem (as in test_bqp.py):
# minimize f + g^T x + 1/2 x^T H x subject to x_l <= x <= x_u, where
#                             ( 1 0 0 )
# f = 1, g = ( 0, 2, 0 ), H = ( 0 2 1 ),
#                             ( 0 1 3 )
# -1 <= x_0 <= 1, x_1 is free and x_2 <= 0
n = 3
infinity = float("inf")
f = 1.0
g = np.array([0.0, 2.0, 0.0])
H = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 3.0]])
x_l = np.array([-1.0, -infinity, -infinity])
x_u = np.array([1.0, infinity, 0.0])

# H is positive definite, so the problem is strictly convex and its
# minimizer is unique. The gradient is H x + g = ( x_0, 2 x_1 + x_2 + 2,
# x_1 + 3 x_2 ), so without bounds x = ( 0, -6/5, 2/5 ), which violates
# x_2 <= 0. With x_2 = 0 on its upper bound, x_0 = 0 and x_1 = -1, and then
# z = H x + g = ( 0, 0, -1 ), with z_2 <= 0 as needed at an upper bound
x_star = np.array([0.0, -1.0, 0.0])
z_star = np.array([0.0, 0.0, -1.0])
f_star = 0.0

STORAGE = ['coordinate', 'sparse_by_rows', 'dense']

def storage(H, H_type):
    # return H_ne, H_row, H_col, H_ptr, H_val for the lower triangle of H
    # in the given storage scheme
    if H_type == 'diagonal':
        return n, None, None, None, np.diag(H).copy()
    rows, cols = np.tril_indices(n)
    if H_type == 'dense':
        return len(rows), None, None, None, H[rows, cols]
    nz = H[rows, cols] != 0.0
    rows, cols = rows[nz], cols[nz]
    if H_type == 'coordinate':
        return len(rows), rows, cols, None, H[rows, cols]
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))]
    return len(rows), None, cols, ptr, H[rows, cols]

def solve(H_type, H, g):
    H_ne, H_row, H_col, H_ptr, H_val = storage(H, H_type)
    options = bqp.initialize()
    options['print_level'] = 0
    bqp.load(n, H_type, H_ne, H_row, H_col, H_ptr, options)
    x, z, x_stat = bqp.solve_qp(n, f, g, H_ne, H_val, x_l, x_u,
                                np.zeros(n), np.zeros(n))
    inform = bqp.information()
    bqp.terminate()
    return x, z, x_stat, inform

def check_solution(H, g, x, z, x_stat, inform, x_star, z_star, f_star):
    # the returned values are consistent, x is feasible and the first-order
    # conditions hold: z = H x + g, with z_j zero, at least zero or at most
    # zero when x_j is free, on its lower bound or on its upper bound. Only
    # x_2 is active, at its upper bound, which x_stat > 0 flags
    assert inform['obj'] == pytest.approx(f + g @ x + 0.5 * x @ H @ x,
                                          abs=1e-8)
    assert np.all(x >= x_l) and np.all(x <= x_u)
    assert z == pytest.approx(H @ x + g, abs=1e-6)
    assert x[2] == pytest.approx(x_u[2], abs=1e-8)
    assert z[:2] == pytest.approx(np.zeros(2), abs=1e-6)
    assert z[2] < 0.0
    assert np.all(x_stat[:2] == 0) and x_stat[2] > 0
    assert x == pytest.approx(x_star, abs=1e-6)
    assert z == pytest.approx(z_star, abs=1e-6)
    assert inform['obj'] == pytest.approx(f_star, abs=1e-8)

@pytest.fixture(params=STORAGE)
def solution(request):
    return solve(request.param, H, g)


def test_bqp_status(solution):
    x, z, x_stat, inform = solution
    assert inform['status'] == 0


def test_bqp_solution(solution):
    x, z, x_stat, inform = solution
    check_solution(H, g, x, z, x_stat, inform, x_star, z_star, f_star)


def test_bqp_diagonal():
    # the diagonal scheme cannot store the H above, so use H = diag(1, 2, 3)
    # and g = ( 0, 2, -3 ) instead. Then without bounds x = ( 0, -1, 1 ),
    # which violates x_2 <= 0, so x_2 = 0 on its upper bound, and as the
    # problem is separable x = ( 0, -1, 0 ), with z = ( 0, 0, -3 ) and
    # f = 1 + 2 x_1 + x_1^2 = 0
    H_d = np.diag([1.0, 2.0, 3.0])
    g_d = np.array([0.0, 2.0, -3.0])
    x_star_d = np.array([0.0, -1.0, 0.0])
    z_star_d = np.array([0.0, 0.0, -3.0])
    f_star_d = 0.0
    x, z, x_stat, inform = solve('diagonal', H_d, g_d)
    assert inform['status'] == 0
    check_solution(H_d, g_d, x, z, x_stat, inform,
                   x_star_d, z_star_d, f_star_d)
