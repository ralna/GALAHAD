from galahad import dqp
import numpy as np
import pytest


# test problems (as in test_dqp.py):
# 1) minimize f + g^T x + 1/2 x^T H x
# subject to c_l <= A x <= c_u and x_l <= x <= x_u, where
#                             ( 1 0 0 )
# f = 1, g = ( 0, 2, 0 ), H = ( 0 2 1 ),
#                             ( 0 1 3 )
# A = ( 2 1 0 ), c_l = ( 1, 2 ), c_u = ( 2, 2 ),
#     ( 0 1 1 )
# -1 <= x_0 <= 1, x_1 is free and x_2 <= 2
# 2) the shifted least-distance problem
# minimize f + g^T x + 1/2 sum_j w_j^2 ( x_j - x^0_j )^2
# subject to the same constraints, with f and g as above, w = ( 1, 1, 1 )
# and x^0 = ( 1, 1, 1 )
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
w = np.array([1.0, 1.0, 1.0])
x0 = np.array([1.0, 1.0, 1.0])

# 1) H is positive definite, so the problem is strictly convex and its
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

# 2) the problem is strictly convex, with gradient W^2 ( x - x^0 ) + g =
# ( x_0 - 1, x_1 + 1, x_2 - 1 ). At x = ( 1, 0, 2 ) this is ( 0, 1, 1 ) =
# A^T y with y = ( 0, 1 ) and z = 0, and x is feasible, so x is the
# minimizer, with f = 1 + 0 + 1/2 ( 0 + 1 + 1 ) = 2. Note that x_0, x_2
# and A_0 x all lie on their upper bounds even though z_0 = z_2 = y_0 = 0,
# so the solution is degenerate
x_star_sld = np.array([1.0, 0.0, 2.0])
y_star_sld = np.array([0.0, 1.0])
z_star_sld = np.zeros(n)
f_star_sld = 2.0

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
    options = dqp.initialize()
    options['print_level'] = 0
    # use the LAPACK solver sytr rather than the defaults, which need the
    # environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    options['symmetric_linear_solver'] = 'sytr'
    options['definite_linear_solver'] = 'sytr'
    options['fdc_options']['symmetric_linear_solver'] = 'sytr'
    options['sbls_options']['symmetric_linear_solver'] = 'sytr'
    options['sbls_options']['definite_linear_solver'] = 'sytr'
    return options

def solve_qp(H_type, H_ne, H_row, H_col, H_ptr, H_val, A_type, g):
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, A_type)
    options = initialize()
    dqp.load(n, m, H_type, H_ne, H_row, H_col, H_ptr,
             A_type, A_ne, A_row, A_col, A_ptr, options)
    x, c, y, z, x_stat, c_stat \
        = dqp.solve_qp(n, m, f, g, H_ne, H_val, A_ne, A_val,
                       c_l, c_u, x_l, x_u, np.zeros(n), np.zeros(m),
                       np.zeros(n))
    inform = dqp.information()
    dqp.terminate()
    return x, c, y, z, x_stat, c_stat, inform

def solve_sldqp(w=w, x0=x0):
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, 'coordinate')
    options = initialize()
    dqp.load(n, m, 'shifted_least_distance', 0, None, None, None,
             'coordinate', A_ne, A_row, A_col, A_ptr, options)
    x, c, y, z, x_stat, c_stat \
        = dqp.solve_sldqp(n, m, f, g, w, x0, A_ne, A_val,
                          c_l, c_u, x_l, x_u, np.zeros(n), np.zeros(m),
                          np.zeros(n))
    inform = dqp.information()
    dqp.terminate()
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


def test_dqp_status(solution):
    x, c, y, z, x_stat, c_stat, inform = solution
    assert inform['status'] == 0


def test_dqp_solution(solution):
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
def test_dqp_separable(case):
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


# dqp requires H to be positive definite, so with H = 0 (the zero and none
# schemes) it should fail with status -20, which the Python interface
# reports as a RuntimeWarning
ZERO = {
    'zero': ('zero', np.zeros(0)),
    'zero-None': ('zero', None),
    'none': ('none', np.zeros(0)),
    'none-None': ('none', None),
}

@pytest.mark.parametrize('case', ZERO)
def test_dqp_zero(case):
    H_type, H_val = ZERO[case]
    with pytest.warns(RuntimeWarning):
        x, c, y, z, x_stat, c_stat, inform \
            = solve_qp(H_type, 0, None, None, None, H_val, 'coordinate', g)
    assert inform['status'] == -20


def test_dqp_sldqp():
    x, c, y, z, x_stat, c_stat, inform = solve_sldqp()
    assert inform['status'] == 0
    assert inform['obj'] == pytest.approx(
        f + g @ x + 0.5 * np.sum(w**2 * (x - x0)**2), abs=1e-8)
    check_optimality(x, c, y, z, w**2 * (x - x0) + g)
    # x_0, x_2 and A_0 x are degenerate, so may or may not be flagged as
    # active, but x_1 is free and the equality A_1 x = 2 is active
    assert x_stat[1] == 0 and c_stat[1] != 0
    assert x == pytest.approx(x_star_sld, abs=1e-8)
    assert y == pytest.approx(y_star_sld, abs=1e-8)
    assert z == pytest.approx(z_star_sld, abs=1e-8)
    assert inform['obj'] == pytest.approx(f_star_sld, abs=1e-8)


# with non-unit weights w = ( 1, 2, 3 ) and shifts x^0 = ( 1, 1, 1 ), the
# Hessian is W^2 = diag( 1, 4, 9 ). The first constraint is active at its
# upper bound 2 x_0 + x_1 = 2, the equality x_1 + x_2 = 2 holds and no
# bounds are active, so solving W^2 ( x - x^0 ) + g = A^T y gives
# x = ( 31, 44, 62 ) / 53 and y = ( -11, 81 ) / 53, with y_0 <= 0 as needed
# at an upper bound, z = 0 and f = 311 / 106. (With w = ( 1, 1, 1 ) as in
# test_dqp.py, w and w^2 cannot be told apart)
def test_dqp_sldqp_weighted():
    w_w = np.array([1.0, 2.0, 3.0])
    x0_w = np.ones(n)
    x, c, y, z, x_stat, c_stat, inform = solve_sldqp(w_w, x0_w)
    assert inform['status'] == 0
    assert inform['obj'] == pytest.approx(
        f + g @ x + 0.5 * np.sum(w_w**2 * (x - x0_w)**2), abs=1e-8)
    check_optimality(x, c, y, z, w_w**2 * (x - x0_w) + g)
    assert x == pytest.approx(np.array([31.0, 44.0, 62.0]) / 53.0, abs=1e-8)
    assert y == pytest.approx(np.array([-11.0, 81.0]) / 53.0, abs=1e-8)
    assert z == pytest.approx(np.zeros(n), abs=1e-8)
    assert inform['obj'] == pytest.approx(311.0 / 106.0, abs=1e-8)
