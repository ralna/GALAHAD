from galahad import lsqp
import numpy as np
import pytest


# test problems: minimize f + g^T x + 1/2 sum_j w_j^2 ( x_j - x^0_j )^2
# subject to c_l <= A x <= c_u and x_l <= x <= x_u, where
# f = 1, A = ( 2 1 0 ), c_l = ( 1, 2 ), c_u = ( 2, 2 ),
#            ( 0 1 1 )
# -1 <= x_0 <= 1, x_1 is free and x_2 <= 2, with
# 1) as in test_lsqp.py, g = ( 0, 2, 0 ), w = ( 1, 1, 1 ), x^0 = ( 1, 1, 1 )
# 2) g = ( 0, 2, 0 ), w = ( 1, 2, 3 ), x^0 = ( 1, 1, 1 )
# 3) the linear program with g = ( 1, 2, 0 ) and w = 0
n = 3
m = 2
infinity = float("inf")
f = 1.0
A = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
c_l = np.array([1.0, 2.0])
c_u = np.array([2.0, 2.0])
x_l = np.array([-1.0, -infinity, -infinity])
x_u = np.array([1.0, infinity, 2.0])

# 1) the gradient W^2 ( x - x^0 ) + g = ( x_0 - 1, x_1 + 1, x_2 - 1 ) is
# ( 0, 1, 1 ) = A^T y at x = ( 1, 0, 2 ), with y = ( 0, 1 ) and z = 0, and
# x is feasible, so x is the (unique) minimizer, with f = 2. But x_0, x_2
# and A_0 x lie on their upper bounds even though z_0 = z_2 = y_0 = 0, and
# lsqp, an interior-point method, converges only slowly to such a
# degenerate solution, so check it to within 1e-5
# 2) W^2 = diag( 1, 4, 9 ). The first constraint is active at its upper
# bound, the equality holds and no bounds are active, so solving
# W^2 ( x - x^0 ) + g = A^T y gives x = ( 31, 44, 62 ) / 53 and
# y = ( -11, 81 ) / 53, with y_0 <= 0 as needed, z = 0 and f = 311 / 106
# 3) g^T x = x_0 + 2 x_1 >= 1/2 + 3/2 x_1 >= 1/2, using 2 x_0 + x_1 >= 1
# and x_1 >= 0 (from x_1 + x_2 = 2 and x_2 <= 2), with equality only at
# x = ( 1/2, 0, 2 ). There g = A^T y + z with y = ( 1/2, 3/2 ) and
# z = ( 0, 0, -3/2 ), and f = 3/2
PROBLEMS = {
    'example': dict(g=np.array([0.0, 2.0, 0.0]), w=np.ones(n), x0=np.ones(n),
                    x=np.array([1.0, 0.0, 2.0]), y=np.array([0.0, 1.0]),
                    z=np.zeros(n), f=2.0, tol=1e-5),
    'weighted': dict(g=np.array([0.0, 2.0, 0.0]), w=np.array([1.0, 2.0, 3.0]),
                     x0=np.ones(n), x=np.array([31.0, 44.0, 62.0]) / 53.0,
                     y=np.array([-11.0, 81.0]) / 53.0, z=np.zeros(n),
                     f=311.0 / 106.0, tol=1e-8),
    'linear': dict(g=np.array([1.0, 2.0, 0.0]), w=np.zeros(n), x0=np.zeros(n),
                   x=np.array([0.5, 0.0, 2.0]), y=np.array([0.5, 1.5]),
                   z=np.array([0.0, 0.0, -1.5]), f=1.5, tol=1e-8),
}

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

def solve(A_type, g, w, x0):
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, A_type)
    options = lsqp.initialize()
    options['print_level'] = 0
    # use the LAPACK solvers sytr and potr rather than the defaults, which
    # need the environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    options['fdc_options']['symmetric_linear_solver'] = 'sytr'
    options['sbls_options']['symmetric_linear_solver'] = 'sytr'
    options['sbls_options']['definite_linear_solver'] = 'potr'
    # lsqp is an interior-point method, so its solution is only as accurate
    # as its stopping tolerances (about 7e-6 by default); tighten them
    options['stop_p'] = options['stop_d'] = options['stop_c'] = 1e-12
    lsqp.load(n, m, A_type, A_ne, A_row, A_col, A_ptr, options)
    x, c, y, z, x_stat, c_stat \
        = lsqp.solve_qp(n, m, f, g, w, x0, A_ne, A_val, c_l, c_u, x_l, x_u,
                        np.zeros(n), np.zeros(m), np.zeros(n))
    inform = lsqp.information()
    lsqp.terminate()
    return x, c, y, z, x_stat, c_stat, inform

def check_optimality(x, c, y, z, grad, tol=1e-8):
    # x is feasible, c = A x, and the first-order conditions hold:
    # grad = A^T y + z, with y_i and z_j zero, at least zero or at most zero
    # when the constraint or bound is inactive, active at its lower bound or
    # active at its upper bound respectively (y_i may have either sign for
    # an equality constraint)
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

@pytest.fixture(params=[(problem, A_type) for problem in PROBLEMS
                        for A_type in A_STORAGE],
                ids=lambda p: f'{p[0]},A={p[1]}')
def solution(request):
    problem, A_type = request.param
    p = PROBLEMS[problem]
    return problem, solve(A_type, p['g'], p['w'], p['x0'])


def test_lsqp_status(solution):
    problem, (x, c, y, z, x_stat, c_stat, inform) = solution
    assert inform['status'] == 0


def test_lsqp_solution(solution):
    problem, (x, c, y, z, x_stat, c_stat, inform) = solution
    p = PROBLEMS[problem]
    g, w, x0, tol = p['g'], p['w'], p['x0'], p['tol']
    assert inform['obj'] == pytest.approx(
        f + g @ x + 0.5 * np.sum(w**2 * (x - x0)**2), abs=1e-8)
    check_optimality(x, c, y, z, w**2 * (x - x0) + g, tol)
    assert x == pytest.approx(p['x'], abs=tol)
    assert y == pytest.approx(p['y'], abs=tol)
    assert z == pytest.approx(p['z'], abs=tol)
    assert inform['obj'] == pytest.approx(p['f'], abs=tol)


def test_lsqp_status_arrays(solution):
    # x_stat and c_stat are negative, zero or positive when a bound or
    # constraint is active at its lower bound, inactive, or active at its
    # upper bound. In every problem x_1 is free and the equality
    # constraint is active
    problem, (x, c, y, z, x_stat, c_stat, inform) = solution
    assert x_stat[1] == 0 and c_stat[1] != 0
    if problem == 'weighted':
        assert x_stat[0] == 0 and x_stat[2] == 0 and c_stat[0] > 0
    elif problem == 'linear':
        assert x_stat[0] == 0 and x_stat[2] > 0 and c_stat[0] < 0
