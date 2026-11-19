from galahad import bqpb
import numpy as np
import pytest


# test problems (as in test_bqpb.py):
# 1) minimize f + g^T x + 1/2 x^T H x subject to x_l <= x <= x_u, where
#                             ( 1 0 0 )
# f = 1, g = ( 0, 2, 0 ), H = ( 0 2 1 ),
#                             ( 0 1 3 )
# -1 <= x_0 <= 1, x_1 is free and x_2 <= 2
# 2) the shifted least-distance problem
# minimize f + g^T x + 1/2 sum_j w_j^2 ( x_j - x^0_j )^2
# subject to the same bounds, with f and g as above, w = ( 1, 1, 1 ) and
# x^0 = ( 1, 1, 1 )
n = 3
infinity = float("inf")
f = 1.0
g = np.array([0.0, 2.0, 0.0])
H = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 3.0]])
x_l = np.array([-1.0, -infinity, -infinity])
x_u = np.array([1.0, infinity, 2.0])
w = np.array([1.0, 1.0, 1.0])
x0 = np.array([1.0, 1.0, 1.0])

# 1) H is positive definite, so the problem is strictly convex and its
# minimizer is unique. The gradient is H x + g = ( x_0, 2 x_1 + x_2 + 2,
# x_1 + 3 x_2 ), which is zero at x = ( 0, -6/5, 2/5 ). This satisfies the
# bounds, so it is the minimizer, with z = 0 and f = 1 + g^T x / 2 = -1/5
x_star = np.array([0.0, -1.2, 0.4])
z_star = np.zeros(n)
f_star = -0.2

# 2) the problem is separable and strictly convex, with gradient
# W^2 ( x - x^0 ) + g = ( x_0 - 1, x_1 + 1, x_2 - 1 ), which is zero at
# x = ( 1, -1, 1 ). This satisfies the bounds, so it is the minimizer, with
# z = 0 and f = 1 - 2 + 1/2 (-2)^2 = 1. Note that x_0 = 1 lies on its
# upper bound even though z_0 = 0, so the solution is degenerate
x_star_sld = np.array([1.0, -1.0, 1.0])
z_star_sld = np.zeros(n)
f_star_sld = 1.0

STORAGE = ['coordinate', 'sparse_by_rows', 'dense']

def storage(H, H_type):
    # return H_ne, H_row, H_col, H_ptr, H_val for the lower triangle of H
    # in the given storage scheme
    rows, cols = np.tril_indices(n)
    if H_type == 'dense':
        return len(rows), None, None, None, H[rows, cols]
    nz = H[rows, cols] != 0.0
    rows, cols = rows[nz], cols[nz]
    if H_type == 'coordinate':
        return len(rows), rows, cols, None, H[rows, cols]
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))]
    return len(rows), None, cols, ptr, H[rows, cols]

def initialize():
    options = bqpb.initialize()
    options['print_level'] = 0
    # use the LAPACK solvers sytr and potr rather than the defaults, which
    # need the environment variables OMP_CANCELLATION and OMP_PROC_BIND set.
    # Without them for sbls_pounce_options, the final "pounce" solve fails
    # and bqpb returns its last interior point, which is only accurate to
    # about 1e-6
    options['fdc_options']['symmetric_linear_solver'] = 'sytr'
    for sbls in ('sbls_options', 'sbls_pounce_options'):
        options[sbls]['symmetric_linear_solver'] = 'sytr'
        options[sbls]['definite_linear_solver'] = 'potr'
    return options

def solve_qp(H_type, H_ne, H_row, H_col, H_ptr, H_val, g, x_u):
    options = initialize()
    bqpb.load(n, H_type, H_ne, H_row, H_col, H_ptr, options)
    x, z, x_stat = bqpb.solve_qp(n, f, g, H_ne, H_val, x_l, x_u,
                                 np.zeros(n), np.zeros(n))
    inform = bqpb.information()
    bqpb.terminate()
    return x, z, x_stat, inform

def solve_sldqp():
    options = initialize()
    bqpb.load(n, 'shifted_least_distance', 0, None, None, None, options)
    x, z, x_stat = bqpb.solve_sldqp(n, f, g, w, x0, x_l, x_u,
                                    np.zeros(n), np.zeros(n))
    inform = bqpb.information()
    bqpb.terminate()
    return x, z, x_stat, inform

def check_optimality(x, z, grad, x_u):
    # x is feasible and the first-order conditions hold: z is the gradient,
    # with z_j zero, at least zero or at most zero when x_j is free, on its
    # lower bound or on its upper bound respectively
    assert np.all(x >= x_l) and np.all(x <= x_u)
    assert z == pytest.approx(grad, abs=1e-8)
    free = (x > x_l + 1e-8) & (x < x_u - 1e-8)
    assert z[free] == pytest.approx(np.zeros(np.sum(free)), abs=1e-8)
    assert np.all(z[x <= x_l + 1e-8] >= -1e-8)
    assert np.all(z[x >= x_u - 1e-8] <= 1e-8)

@pytest.fixture(params=STORAGE)
def solution(request):
    return solve_qp(request.param, *storage(H, request.param), g, x_u)


def test_bqpb_status(solution):
    x, z, x_stat, inform = solution
    assert inform['status'] == 0


def test_bqpb_solution(solution):
    x, z, x_stat, inform = solution
    assert inform['obj'] == pytest.approx(f + g @ x + 0.5 * x @ H @ x,
                                          abs=1e-8)
    check_optimality(x, z, H @ x + g, x_u)
    assert np.all(x_stat == 0)
    assert x == pytest.approx(x_star, abs=1e-8)
    assert z == pytest.approx(z_star, abs=1e-8)
    assert inform['obj'] == pytest.approx(f_star, abs=1e-8)


# the diagonal, scaled_identity and identity schemes cannot store the H
# above, so solve the separable problem with H = diag(d), g = ( 0, 2, -3 )
# and x_2 <= 0 instead, given H_type, H_ne and H_val for each case. There
# are no values to give for the identity, so H_val may be empty or None
SEPARABLE = {
    'diagonal': ('diagonal', np.array([1.0, 2.0, 3.0]), 3,
                 np.array([1.0, 2.0, 3.0])),
    'scaled_identity': ('scaled_identity', np.array([2.0, 2.0, 2.0]), 1,
                        np.array([2.0])),
    'identity': ('identity', np.array([1.0, 1.0, 1.0]), 0, np.zeros(0)),
    'identity-None': ('identity', np.array([1.0, 1.0, 1.0]), 0, None),
}

@pytest.mark.parametrize('case', SEPARABLE)
def test_bqpb_separable(case):
    # each x_j minimizes g_j x_j + d_j x_j^2 / 2 within its bounds. Without
    # bounds x = ( 0, -2 / d_1, 3 / d_2 ), and the last violates x_2 <= 0, so
    # x = ( 0, -2 / d_1, 0 ), z = ( 0, 0, -3 ) and f = 1 - 2 / d_1
    H_type, d, H_ne, H_val = SEPARABLE[case]
    g_s = np.array([0.0, 2.0, -3.0])
    x_u_s = np.array([1.0, infinity, 0.0])
    x_star_s = np.array([0.0, -2.0 / d[1], 0.0])
    z_star_s = np.array([0.0, 0.0, -3.0])
    f_star_s = 1.0 - 2.0 / d[1]
    x, z, x_stat, inform = solve_qp(H_type, H_ne, None, None, None, H_val,
                                    g_s, x_u_s)
    assert inform['status'] == 0
    assert inform['obj'] == pytest.approx(f + g_s @ x + 0.5 * d @ x**2,
                                          abs=1e-8)
    check_optimality(x, z, d * x + g_s, x_u_s)
    assert np.all(x_stat[:2] == 0) and x_stat[2] > 0
    assert x == pytest.approx(x_star_s, abs=1e-8)
    assert z == pytest.approx(z_star_s, abs=1e-8)
    assert inform['obj'] == pytest.approx(f_star_s, abs=1e-8)


def test_bqpb_sldqp():
    x, z, x_stat, inform = solve_sldqp()
    assert inform['status'] == 0
    assert inform['obj'] == pytest.approx(
        f + g @ x + 0.5 * np.sum(w**2 * (x - x0)**2), abs=1e-8)
    check_optimality(x, z, w**2 * (x - x0) + g, x_u)
    # x_0 is degenerate, so it may or may not be flagged as active
    assert x_stat[0] >= 0 and np.all(x_stat[1:] == 0)
    assert x == pytest.approx(x_star_sld, abs=1e-8)
    assert z == pytest.approx(z_star_sld, abs=1e-8)
    assert inform['obj'] == pytest.approx(f_star_sld, abs=1e-8)
