from galahad import qpb
import numpy as np
import pytest


# test problem (as in test_qpb.py):
# minimize f + g^T x + 1/2 x^T H x
# subject to c_l <= A x <= c_u and x_l <= x <= x_u, where
#                             ( 1 0 0 )
# f = 1, g = ( 0, 2, 0 ), H = ( 0 2 1 ),
#                             ( 0 1 3 )
# A = ( 2 1 0 ), c_l = ( 1, 2 ), c_u = ( 2, 2 ),
#     ( 0 1 1 )
# -1 <= x_0 <= 1, x_1 is free and x_2 <= 2
n = 3
m = 2
infinity = float("inf")
f = 1.0
g = np.array([0.0, 2.0, 0.0])
H = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 3.0]])
A = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
c_l = np.array([1.0, 2.0])
c_u = np.array([2.0, 2.0])
x_l = np.array([-1.0, -infinity, -infinity])
x_u = np.array([1.0, infinity, 2.0])

# H is positive definite, so the problem is strictly convex and its
# minimizer is unique. Both constraints are active at their lower bounds
# and no bounds are active: solving H x + g = A^T y, 2 x_0 + x_1 = 1 and
# x_1 + x_2 = 2 gives x = ( 2, 9, 17 ) / 13 and y = ( 1, 60 ) / 13. Then x
# satisfies its bounds, y_0 >= 0 as needed at a lower bound (y_1 may have
# either sign as c_l = c_u for the second constraint), z = 0 and
# f = 165 / 26
x_star = np.array([2.0, 9.0, 17.0]) / 13.0
y_star = np.array([1.0, 60.0]) / 13.0
z_star = np.zeros(n)
f_star = 165.0 / 26.0

H_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']
A_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']

def h_storage(H, H_type):
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
    options = qpb.initialize()
    options['print_level'] = 0
    # use the LAPACK solvers sytr and potr rather than the defaults, which
    # need the environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    for opts in (options, options['lsqp_options']):
        opts['fdc_options']['symmetric_linear_solver'] = 'sytr'
        opts['sbls_options']['symmetric_linear_solver'] = 'sytr'
        opts['sbls_options']['definite_linear_solver'] = 'potr'
    # qpb is an interior-point method, so its solution is only as accurate
    # as its stopping tolerances (stop_p, stop_d and stop_c, about 7e-6 by
    # default); tighten them so that the solution is accurate to 1e-8
    options['stop_p'] = options['stop_d'] = options['stop_c'] = 1e-12
    return options

def solve_qp(H_type, H_ne, H_row, H_col, H_ptr, H_val, A_type, g):
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, A_type)
    options = initialize()
    qpb.load(n, m, H_type, H_ne, H_row, H_col, H_ptr,
             A_type, A_ne, A_row, A_col, A_ptr, options)
    x, c, y, z, x_stat, c_stat \
        = qpb.solve_qp(n, m, f, g, H_ne, H_val, A_ne, A_val,
                       c_l, c_u, x_l, x_u, np.zeros(n), np.zeros(m),
                       np.zeros(n))
    inform = qpb.information()
    qpb.terminate()
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

@pytest.fixture(params=[(H_type, A_type) for H_type in H_STORAGE
                        for A_type in A_STORAGE],
                ids=lambda p: f'H={p[0]},A={p[1]}')
def solution(request):
    H_type, A_type = request.param
    return solve_qp(H_type, *h_storage(H, H_type), A_type, g)


def test_qpb_status(solution):
    x, c, y, z, x_stat, c_stat, inform = solution
    assert inform['status'] == 0


def test_qpb_solution(solution):
    x, c, y, z, x_stat, c_stat, inform = solution
    assert inform['obj'] == pytest.approx(f + g @ x + 0.5 * x @ H @ x,
                                          abs=1e-8)
    check_optimality(x, c, y, z, H @ x + g)
    # c_stat < 0 flags the constraints active at their lower bounds
    assert np.all(x_stat == 0) and np.all(c_stat < 0)
    assert x == pytest.approx(x_star, abs=1e-8)
    assert y == pytest.approx(y_star, abs=1e-8)
    assert z == pytest.approx(z_star, abs=1e-8)
    assert inform['obj'] == pytest.approx(f_star, abs=1e-8)


# the diagonal, scaled_identity and identity schemes cannot store the H
# above, so solve the problem with H = diag(d) instead, given H_type, H_ne
# and H_val for each case. There are no values to give for the identity,
# so H_val may be empty or None. As for H above, both constraints are
# active at their lower bounds and no bounds are active, and solving
# diag(d) x + g = A^T y, 2 x_0 + x_1 = 1 and x_1 + x_2 = 2 gives x, y and f
SEPARABLE = {
    'diagonal': ('diagonal', np.array([1.0, 2.0, 3.0]), 3,
                 np.array([1.0, 2.0, 3.0]), np.array([2.0, 17.0, 25.0]) / 21.0,
                 np.array([1.0, 75.0]) / 21.0, 227.0 / 42.0),
    'scaled_identity': ('scaled_identity', np.array([2.0, 2.0, 2.0]), 1,
                        np.array([2.0]), np.array([2.0, 5.0, 13.0]) / 9.0,
                        np.array([2.0, 26.0]) / 9.0, 41.0 / 9.0),
    'identity': ('identity', np.array([1.0, 1.0, 1.0]), 0, np.zeros(0),
                 np.array([4.0, 1.0, 17.0]) / 9.0, np.array([2.0, 17.0]) / 9.0,
                 28.0 / 9.0),
    'identity-None': ('identity', np.array([1.0, 1.0, 1.0]), 0, None,
                      np.array([4.0, 1.0, 17.0]) / 9.0,
                      np.array([2.0, 17.0]) / 9.0, 28.0 / 9.0),
}

@pytest.mark.parametrize('case', SEPARABLE)
def test_qpb_separable(case):
    H_type, d, H_ne, H_val, x_star_s, y_star_s, f_star_s = SEPARABLE[case]
    x, c, y, z, x_stat, c_stat, inform \
        = solve_qp(H_type, H_ne, None, None, None, H_val, 'coordinate', g)
    assert inform['status'] == 0
    assert inform['obj'] == pytest.approx(f + g @ x + 0.5 * d @ x**2,
                                          abs=1e-8)
    check_optimality(x, c, y, z, d * x + g)
    assert np.all(x_stat == 0) and np.all(c_stat < 0)
    assert x == pytest.approx(x_star_s, abs=1e-8)
    assert y == pytest.approx(y_star_s, abs=1e-8)
    assert z == pytest.approx(np.zeros(n), abs=1e-8)
    assert inform['obj'] == pytest.approx(f_star_s, abs=1e-8)


# with H = 0 (the zero and none schemes), the problem is a linear program,
# and g = ( 0, 2, 0 ) would give a non-unique minimizer, so use
# g = ( 1, 2, 0 ) instead. As x_1 + x_2 = 2 and x_2 <= 2, x_1 >= 0, and
# 2 x_0 + x_1 >= 1 gives g^T x = x_0 + 2 x_1 >= 1/2 + 3/2 x_1 >= 1/2, with
# equality only at the vertex x = ( 1/2, 0, 2 ). There g = A^T y + z with
# y = ( 1/2, 3/2 ) and z = ( 0, 0, -3/2 ), with z_2 <= 0 as needed at an
# upper bound, and f = 3/2
LINEAR = {
    'zero': ('zero', np.zeros(0)),
    'zero-None': ('zero', None),
    'none': ('none', np.zeros(0)),
    'none-None': ('none', None),
}

@pytest.mark.parametrize('case', LINEAR)
def test_qpb_linear(case):
    H_type, H_val = LINEAR[case]
    g_lp = np.array([1.0, 2.0, 0.0])
    x, c, y, z, x_stat, c_stat, inform \
        = solve_qp(H_type, 0, None, None, None, H_val, 'coordinate', g_lp)
    assert inform['status'] == 0
    assert inform['obj'] == pytest.approx(f + g_lp @ x, abs=1e-8)
    check_optimality(x, c, y, z, g_lp)
    assert np.all(x_stat[:2] == 0) and x_stat[2] > 0 and np.all(c_stat < 0)
    assert x == pytest.approx(np.array([0.5, 0.0, 2.0]), abs=1e-8)
    assert y == pytest.approx(np.array([0.5, 1.5]), abs=1e-8)
    assert z == pytest.approx(np.array([0.0, 0.0, -1.5]), abs=1e-8)
    assert inform['obj'] == pytest.approx(1.5, abs=1e-8)


def test_qpb_dependency_failure():
    # ask fdc, which qpb uses to find dependent equality constraints, for
    # an unknown linear solver, so that it fails with status -26
    H_ne, H_row, H_col, H_ptr, H_val = h_storage(H, 'coordinate')
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, 'coordinate')
    options = initialize()
    options['fdc_options']['symmetric_linear_solver'] = 'unknown'
    qpb.load(n, m, 'coordinate', H_ne, H_row, H_col, H_ptr,
             'coordinate', A_ne, A_row, A_col, A_ptr, options)
    with pytest.warns(RuntimeWarning):
        qpb.solve_qp(n, m, f, g, H_ne, H_val, A_ne, A_val, c_l, c_u, x_l,
                     x_u, np.zeros(n), np.zeros(m), np.zeros(n))
    inform = qpb.information()
    qpb.terminate()
    assert inform['fdc_inform']['status'] == -26
    assert inform['status'] == -26
