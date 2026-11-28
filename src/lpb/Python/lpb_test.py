from galahad import lpb
import numpy as np
import pytest


# test problems:
# 1) as in test_lpb.py, minimize f + g^T x
# subject to c_l <= A x <= c_u and x_l <= x <= x_u, where
# f = 1, g = ( 0, 2, 0 ), A = ( 2 1 0 ), c_l = ( 1, 2 ), c_u = ( 2, 2 ),
#                             ( 0 1 1 )
# -1 <= x_0 <= 1, x_1 is free and x_2 <= 2
# 2) the same with g = ( 1, 2, 0 )
n = 3
m = 2
infinity = float("inf")
f = 1.0
g = np.array([0.0, 2.0, 0.0])
A = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
c_l = np.array([1.0, 2.0])
c_u = np.array([2.0, 2.0])
x_l = np.array([-1.0, -infinity, -infinity])
x_u = np.array([1.0, infinity, 2.0])

# 1) as x_1 + x_2 = 2 and x_2 <= 2, x_1 >= 0, so f + g^T x = 1 + 2 x_1 >= 1,
# with equality exactly when x_1 = 0 and x_2 = 2, and then 2 x_0 + x_1 in
# [1, 2] requires x_0 in [1/2, 1]. So the minimizers are not unique, but the
# optimal value is 1 (lpb, an interior-point method, finds a point in the
# interior of this face)
f_star = 1.0

# 2) g^T x = x_0 + 2 x_1 >= 1/2 + 3/2 x_1 >= 1/2, using 2 x_0 + x_1 >= 1
# and x_1 >= 0, with equality only at the vertex x = ( 1/2, 0, 2 ). There
# g = A^T y + z with y = ( 1/2, 3/2 ) and z = ( 0, 0, -3/2 ), and f = 3/2
g_2 = np.array([1.0, 2.0, 0.0])
x_star_2 = np.array([0.5, 0.0, 2.0])
y_star_2 = np.array([0.5, 1.5])
z_star_2 = np.array([0.0, 0.0, -1.5])
f_star_2 = 1.5

A_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']

def a_storage(A, A_type):
    # return A_ne, A_row, A_col, A_ptr, A_val for A in the given storage
    # scheme
    if A_type == 'dense':
        return A.size, None, None, None, A.ravel()
    rows, cols = np.nonzero(A)
    if A_type == 'coordinate':
        return len(rows), rows, cols, None, A[rows, cols]
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=m))]
    return len(rows), None, cols, ptr, A[rows, cols]

def initialize():
    options = lpb.initialize()
    options['print_level'] = 0
    # use the LAPACK solvers sytr and potr rather than the defaults, which
    # need the environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    options['fdc_options']['symmetric_linear_solver'] = 'sytr'
    options['sbls_options']['symmetric_linear_solver'] = 'sytr'
    options['sbls_options']['definite_linear_solver'] = 'potr'
    # lpb is an interior-point method, so its solution is only as accurate
    # as its stopping tolerances (about 7e-6 by default); tighten them so
    # that the solution is accurate to 1e-8
    options['stop_abs_p'] = options['stop_abs_d'] = 1e-12
    options['stop_abs_c'] = 1e-12
    return options

def solve(A_type, g, options=None):
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, A_type)
    if options is None:
        options = initialize()
    lpb.load(n, m, A_type, A_ne, A_row, A_col, A_ptr, options)
    x, c, y, z, x_stat, c_stat \
        = lpb.solve_lp(n, m, f, g, A_ne, A_val, c_l, c_u, x_l, x_u,
                       np.zeros(n), np.zeros(m), np.zeros(n))
    inform = lpb.information()
    lpb.terminate()
    return x, c, y, z, x_stat, c_stat, inform

def check_optimality(x, c, y, z, grad):
    # x is feasible, c = A x, and the first-order conditions hold:
    # grad = A^T y + z, with y_i and z_j zero, at least zero or at most zero
    # when the constraint or bound is inactive, active at its lower bound or
    # active at its upper bound respectively (y_i may have either sign for
    # an equality constraint)
    tol = 1e-8
    assert c == pytest.approx(A @ x, abs=tol)
    assert np.all(x >= x_l - tol) and np.all(x <= x_u + tol)
    assert np.all(c >= c_l - tol) and np.all(c <= c_u + tol)
    assert grad == pytest.approx(A.T @ y + z, abs=tol)
    at_l, at_u = x <= x_l + tol, x >= x_u - tol
    assert z[~at_l & ~at_u] == pytest.approx(np.zeros(np.sum(~at_l & ~at_u)),
                                             abs=tol)
    assert np.all(z[at_l & ~at_u] >= -tol)
    assert np.all(z[at_u & ~at_l] <= tol)
    at_l, at_u = c <= c_l + tol, c >= c_u - tol
    assert y[~at_l & ~at_u] == pytest.approx(np.zeros(np.sum(~at_l & ~at_u)),
                                             abs=tol)
    assert np.all(y[at_l & ~at_u] >= -tol)
    assert np.all(y[at_u & ~at_l] <= tol)

@pytest.mark.parametrize('A_type', A_STORAGE)
def test_lpb_example(A_type):
    x, c, y, z, x_stat, c_stat, inform = solve(A_type, g)
    assert inform['status'] == 0
    assert inform['obj'] == pytest.approx(f + g @ x, abs=1e-8)
    check_optimality(x, c, y, z, g)
    assert x[1:] == pytest.approx(np.array([0.0, 2.0]), abs=1e-8)
    assert 0.5 - 1e-8 <= x[0] <= 1.0 + 1e-8
    assert inform['obj'] == pytest.approx(f_star, abs=1e-8)


@pytest.mark.parametrize('A_type', A_STORAGE)
def test_lpb_vertex(A_type):
    x, c, y, z, x_stat, c_stat, inform = solve(A_type, g_2)
    assert inform['status'] == 0
    assert inform['obj'] == pytest.approx(f + g_2 @ x, abs=1e-8)
    check_optimality(x, c, y, z, g_2)
    # x_2 is active at its upper bound, and both constraints at their
    # lower bounds
    assert np.all(x_stat[:2] == 0) and x_stat[2] > 0 and np.all(c_stat < 0)
    assert x == pytest.approx(x_star_2, abs=1e-8)
    assert y == pytest.approx(y_star_2, abs=1e-8)
    assert z == pytest.approx(z_star_2, abs=1e-8)
    assert inform['obj'] == pytest.approx(f_star_2, abs=1e-8)


def test_lpb_getdua():
    # advanced initial values for the dual variables should give the same
    # solution
    options = initialize()
    options['getdua'] = True
    x, c, y, z, x_stat, c_stat, inform = solve('coordinate', g_2, options)
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_2, abs=1e-8)
    assert inform['obj'] == pytest.approx(f_star_2, abs=1e-8)
