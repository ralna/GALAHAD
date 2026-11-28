from galahad import qpa
import numpy as np
import pytest


# test problems (as in test_qpa.py):
# 1) minimize q(x) = f + g^T x + 1/2 x^T H x
# subject to c_l <= A x <= c_u and x_l <= x <= x_u, where
#                             ( 1 0 0 )
# f = 1, g = ( 0, 2, 0 ), H = ( 0 2 1 ),
#                             ( 0 1 3 )
# A = ( 2 1 0 ), c_l = ( 1, 2 ), c_u = ( 2, 2 ),
#     ( 0 1 1 )
# -1 <= x_0 <= 1, x_1 is free and x_2 <= 2
# 2) the l_1 quadratic program
# minimize q(x) + rho_g v_g(x) + rho_b v_b(x), where
# v_g(x) = sum_i max( c_l_i - a_i^T x, 0 ) + max( a_i^T x - c_u_i, 0 ) and
# v_b(x) = sum_j max( x_l_j - x_j, 0 ) + max( x_j - x_u_j, 0 )
# 3) the bound-constrained l_1 quadratic program
# minimize q(x) + rho_g v_g(x) subject to x_l <= x <= x_u
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
    options = qpa.initialize()
    options['print_level'] = 0
    # use the LAPACK solver sytr rather than the default, which needs the
    # environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    options['symmetric_linear_solver'] = 'sytr'
    return options

def solve_qp(H_type, H_ne, H_row, H_col, H_ptr, H_val, A_type, g):
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, A_type)
    options = initialize()
    qpa.load(n, m, H_type, H_ne, H_row, H_col, H_ptr,
             A_type, A_ne, A_row, A_col, A_ptr, options)
    x, c, y, z, x_stat, c_stat \
        = qpa.solve_qp(n, m, f, g, H_ne, H_val, A_ne, A_val,
                       c_l, c_u, x_l, x_u, np.zeros(n), np.zeros(m),
                       np.zeros(n))
    inform = qpa.information()
    qpa.terminate()
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


def test_qpa_status(solution):
    x, c, y, z, x_stat, c_stat, inform = solution
    assert inform['status'] == 0


def test_qpa_solution(solution):
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
def test_qpa_separable(case):
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
def test_qpa_linear(case):
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


# the l_1 problems. With c_u = ( 2, 2 ) as in test_qpa.py, the second
# constraint is an equality, which qpa enforces exactly rather than
# penalising (see test_qpa_l1qp_equality below), so make it the inequality
# 2 <= x_1 + x_2 <= 3 instead. The penalty rho_g = 1 is smaller than the
# multiplier 60/13 of the second constraint at the solution of problem 1,
# so the l_1 minimizers violate it
c_u_l1 = np.array([2.0, 3.0])

def solve_l1(kind, x_u, rho_g, rho_b=None):
    H_ne, H_row, H_col, H_ptr, H_val = h_storage(H, 'coordinate')
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, 'coordinate')
    options = initialize()
    qpa.load(n, m, 'coordinate', H_ne, H_row, H_col, H_ptr,
             'coordinate', A_ne, A_row, A_col, A_ptr, options)
    args = (np.zeros(n), np.zeros(m), np.zeros(n))
    if kind == 'l1qp':
        x, c, y, z, x_stat, c_stat \
            = qpa.solve_l1qp(n, m, f, g, H_ne, H_val, rho_g, rho_b, A_ne,
                             A_val, c_l, c_u_l1, x_l, x_u, *args)
    else:
        x, c, y, z, x_stat, c_stat \
            = qpa.solve_bcl1qp(n, m, f, g, H_ne, H_val, rho_g, A_ne, A_val,
                               c_l, c_u_l1, x_l, x_u, *args)
    inform = qpa.information()
    qpa.terminate()
    return x, c, y, z, x_stat, c_stat, inform

def violations(v, v_l, v_u):
    # the l_1 infeasibility
    # sum_i max( v_l_i - v_i, 0 ) + max( v_i - v_u_i, 0 )
    return np.sum(np.maximum(v_l - v, 0.0) + np.maximum(v - v_u, 0.0))

def check_l1_optimality(x, c, y, z, x_u, rho_g, rho_b):
    # c = A x, and the first-order conditions for the l_1 problem hold:
    # g + H x = A^T y + z, where y_i is rho_g or -rho_g if the i-th
    # constraint is violated below or above, zero if it is strictly
    # satisfied, and lies in [0, rho_g] or [-rho_g, 0] if it is active at its
    # lower or upper bound. Likewise for z_j, with rho_b, or with x feasible
    # and z_j unbounded at an active bound if rho_b is None (bound-constrained)
    tol = 1e-8
    assert c == pytest.approx(A @ x, abs=tol)
    assert g + H @ x == pytest.approx(A.T @ y + z, abs=tol)
    def check(v, v_l, v_u, mult, rho):
        below, above = v < v_l - tol, v > v_u + tol
        at_l = ~below & (v <= v_l + tol)
        at_u = ~above & (v >= v_u - tol)
        inside = ~below & ~above & ~at_l & ~at_u
        assert mult[below] == pytest.approx(rho * np.ones(np.sum(below)),
                                            abs=tol)
        assert mult[above] == pytest.approx(-rho * np.ones(np.sum(above)),
                                            abs=tol)
        assert mult[inside] == pytest.approx(np.zeros(np.sum(inside)),
                                             abs=tol)
        assert np.all(mult[at_l & ~at_u] >= -tol)
        assert np.all(mult[at_u & ~at_l] <= tol)
        assert np.all(mult[at_l | at_u] <= rho + tol)
        assert np.all(mult[at_l | at_u] >= -rho - tol)
    check(c, c_l, c_u_l1, y, rho_g)
    if rho_b is None:
        assert np.all(x >= x_l - tol) and np.all(x <= x_u + tol)
        check(x, x_l, x_u, z, np.inf)
    else:
        check(x, x_l, x_u, z, rho_b)

# for each case, the KKT conditions above, with the stated constraints
# active or violated, give x, y and z, and the inform values
#   obj = q(x), infeas_g = v_g(x), infeas_b = v_b(x) and
#   merit = q(x) + rho_g v_g(x) + rho_b v_b(x)
# (with rho_b = 0 for the bound-constrained problem)
L1 = {
    # A_0 x = 1 is active at its lower bound, A_1 x = -1/23 violates its
    # lower bound, and no bounds are active, so y = ( 9/23, 1 ) and z = 0
    'l1qp': ('l1qp', 1.0, 1.0, 1.0, np.array([18.0, -13.0, 12.0]) / 23.0,
             np.array([9.0 / 23.0, 1.0]), np.zeros(n),
             14.0 / 23.0, 47.0 / 23.0, 0.0, 61.0 / 23.0),
    # the bounds are inactive at the l1qp solution, so the bound-constrained
    # problem has the same solution
    'bcl1qp': ('bcl1qp', 1.0, 1.0, None, np.array([18.0, -13.0, 12.0]) / 23.0,
               np.array([9.0 / 23.0, 1.0]), np.zeros(n),
               14.0 / 23.0, 47.0 / 23.0, 0.0, 61.0 / 23.0),
    # with x_0 <= 1/2 and rho_b = 1/10, the bound on x_0 is also violated,
    # so z_0 = -1/10, and then y = ( 10/23, 1 )
    'l1qp-x_u_0=1/2': ('l1qp', 0.5, 1.0, 0.1,
                       np.array([177.0 / 230.0, -62.0 / 115.0, 59.0 / 115.0]),
                       np.array([10.0 / 23.0, 1.0]),
                       np.array([-0.1, 0.0, 0.0]), 2883.0 / 4600.0,
                       233.0 / 115.0, 31.0 / 115.0, 12327.0 / 4600.0),
}

@pytest.mark.parametrize('case', L1)
def test_qpa_l1(case):
    kind, x_u_0, rho_g, rho_b, x_star_l1, y_star_l1, z_star_l1, \
        obj_star, infeas_g_star, infeas_b_star, merit_star = L1[case]
    x_u_l1 = np.r_[x_u_0, x_u[1:]]
    x, c, y, z, x_stat, c_stat, inform = solve_l1(kind, x_u_l1, rho_g, rho_b)
    assert inform['status'] == 0
    check_l1_optimality(x, c, y, z, x_u_l1, rho_g, rho_b)
    assert inform['obj'] == pytest.approx(f + g @ x + 0.5 * x @ H @ x,
                                          abs=1e-8)
    assert inform['infeas_g'] == pytest.approx(violations(c, c_l, c_u_l1),
                                               abs=1e-8)
    assert inform['infeas_b'] == pytest.approx(violations(x, x_l, x_u_l1),
                                               abs=1e-8)
    assert x == pytest.approx(x_star_l1, abs=1e-8)
    assert y == pytest.approx(y_star_l1, abs=1e-8)
    assert z == pytest.approx(z_star_l1, abs=1e-8)
    assert inform['obj'] == pytest.approx(obj_star, abs=1e-8)
    assert inform['infeas_g'] == pytest.approx(infeas_g_star, abs=1e-8)
    assert inform['infeas_b'] == pytest.approx(infeas_b_star, abs=1e-8)
    assert inform['merit'] == pytest.approx(merit_star, abs=1e-8)


def test_qpa_bcl1qp_active_bound():
    # with x_0 <= 1/2, the l_1 minimizer above violates the bound, so it
    # must be active: x_0 = 1/2, both constraints are violated below, so
    # y = ( 1, 1 ), and then x = ( 1/2, -1/5, 2/5 ), z = ( -3/2, 0, 0 ),
    # q(x) = 37/40, v_g(x) = 2 and the merit function is 117/40
    x_u_l1 = np.r_[0.5, x_u[1:]]
    x, c, y, z, x_stat, c_stat, inform = solve_l1('bcl1qp', x_u_l1, 1.0)
    assert inform['status'] == 0
    check_l1_optimality(x, c, y, z, x_u_l1, 1.0, None)
    assert x == pytest.approx(np.array([0.5, -0.2, 0.4]), abs=1e-8)
    assert y == pytest.approx(np.array([1.0, 1.0]), abs=1e-8)
    assert z == pytest.approx(np.array([-1.5, 0.0, 0.0]), abs=1e-8)
    assert inform['merit'] == pytest.approx(117.0 / 40.0, abs=1e-8)


@pytest.mark.xfail(strict=True, reason='qpa enforces equality constraints '
                   'exactly in its l_1 modes rather than penalising them '
                   '(see https://github.com/ralna/GALAHAD/issues/610)')
def test_qpa_l1qp_equality():
    # with c_u = ( 2, 2 ) as in test_qpa.py, the second constraint is the
    # equality x_1 + x_2 = 2. Its l_1 objective is at least that with
    # c_u = ( 2, 3 ) everywhere, with equality wherever x_1 + x_2 <= 2,
    # which holds at the l1qp minimizer above (x_1 + x_2 = -1/23), so that
    # is also the minimizer here. qpa instead returns the solution of
    # problem 1
    H_ne, H_row, H_col, H_ptr, H_val = h_storage(H, 'coordinate')
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, 'coordinate')
    options = initialize()
    qpa.load(n, m, 'coordinate', H_ne, H_row, H_col, H_ptr,
             'coordinate', A_ne, A_row, A_col, A_ptr, options)
    x, c, y, z, x_stat, c_stat \
        = qpa.solve_l1qp(n, m, f, g, H_ne, H_val, 1.0, 1.0, A_ne, A_val,
                         c_l, c_u, x_l, x_u, np.zeros(n), np.zeros(m),
                         np.zeros(n))
    inform = qpa.information()
    qpa.terminate()
    assert inform['status'] == 0
    assert x == pytest.approx(np.array([18.0, -13.0, 12.0]) / 23.0, abs=1e-8)
    assert y == pytest.approx(np.array([9.0 / 23.0, 1.0]), abs=1e-8)
    assert inform['merit'] == pytest.approx(61.0 / 23.0, abs=1e-8)
