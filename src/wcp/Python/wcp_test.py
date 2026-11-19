from galahad import wcp
import numpy as np
import pytest


# wcp finds a well-centred point x of c_l <= A x <= c_u, x_l <= x <= x_u,
# together with c = A x and duals y_l, y_u, z_l and z_u for which
# (3) A x = c, (4) g = A^T ( y_l + y_u ) + z_l + z_u, (5)-(6) each
# ( c_i - c^l_i ) y^l_i, ( c_i - c^u_i ) y^u_i, ( x_j - x^l_j ) z^l_j and
# ( x_j - x^u_j ) z^u_j equals a strictly positive target, and (7) the
# bounds hold with y_l, z_l >= 0 and y_u, z_u <= 0. The targets are chosen
# (and adjusted) internally, so the point is not known in advance; check
# (3)-(7) instead, and the exact point for a box, where symmetry fixes it.
#
# test problems, with A = ( 2 1 0 ) unless stated otherwise:
#                         ( 0 1 1 )
# 1) as in test_wcp.py, c_l = ( 1, 2 ), c_u = ( 2, 2 ), -1 <= x_0 <= 1,
# x_1 free and x_2 <= 2, g = 0. The equality constraint means there is no
# strict interior, so the point is not strictly feasible
# 2) the strictly feasible c_l = ( 1, 1 ), c_u = ( 2, 3 ), -1 <= x_0 <= 1,
# -10 <= x_1 <= 10 and x_2 <= 2, with g = 0
# 3) as 2) with g = ( 1, -1, 2 )
# 4) as 2) with x_1 free
# 5) the box -1 <= x_0 <= 1, 0 <= x_1 <= 4 and 2 <= x_2 <= 3, with no
# general constraints and g = 0. Then z_l = - z_u, so the targets for the
# two bounds on x_j are equal, ( x_j - x^l_j ) z^l_j = ( x^u_j - x_j ) z^l_j,
# and x is the centre ( 0, 2, 5/2 ) of the box
infinity = float("inf")
A_EXAMPLE = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
PROBLEMS = {
    'example': dict(A=A_EXAMPLE, c_l=np.array([1.0, 2.0]),
                    c_u=np.array([2.0, 2.0]),
                    x_l=np.array([-1.0, -infinity, -infinity]),
                    x_u=np.array([1.0, infinity, 2.0]), g=np.zeros(3),
                    feasible=False),
    'interior': dict(A=A_EXAMPLE, c_l=np.array([1.0, 1.0]),
                     c_u=np.array([2.0, 3.0]),
                     x_l=np.array([-1.0, -10.0, -infinity]),
                     x_u=np.array([1.0, 10.0, 2.0]), g=np.zeros(3),
                     feasible=True),
    'target': dict(A=A_EXAMPLE, c_l=np.array([1.0, 1.0]),
                   c_u=np.array([2.0, 3.0]),
                   x_l=np.array([-1.0, -10.0, -infinity]),
                   x_u=np.array([1.0, 10.0, 2.0]),
                   g=np.array([1.0, -1.0, 2.0]), feasible=True),
    'free': dict(A=A_EXAMPLE, c_l=np.array([1.0, 1.0]),
                 c_u=np.array([2.0, 3.0]),
                 x_l=np.array([-1.0, -infinity, -infinity]),
                 x_u=np.array([1.0, infinity, 2.0]), g=np.zeros(3),
                 feasible=True),
    'box': dict(A=np.zeros((0, 3)), c_l=np.zeros(0), c_u=np.zeros(0),
                x_l=np.array([-1.0, 0.0, 2.0]), x_u=np.array([1.0, 4.0, 3.0]),
                g=np.zeros(3), feasible=True),
}
x_star_box = np.array([0.0, 2.0, 2.5])

# wcp counts the zero dual z_l of a free variable as an implicit (always
# inactive) bound, so flags a free variable as x_stat = -2 and a strictly
# feasible point as not feasible
FREE_XFAIL = pytest.mark.xfail(
    strict=True, reason='wcp treats free variables as implicit bounds '
    '(see https://github.com/ralna/GALAHAD/pull/616)')
# wcp drops the multipliers of equality constraints, returning y_l = y_u = 0
EQUALITY_XFAIL = pytest.mark.xfail(
    strict=True, reason='wcp does not return equality multipliers '
    '(see https://github.com/ralna/GALAHAD/issues/617)')

A_STORAGE = ['coordinate', 'sparse_by_rows', 'dense']

def a_storage(A, A_type):
    # return A_ne, A_row, A_col, A_ptr, A_val for A in the given storage
    # scheme
    if A_type == 'dense':
        return A.size, None, None, None, A.ravel()
    rows, cols = np.nonzero(A)
    if A_type == 'coordinate':
        return len(rows), rows, cols, None, A[rows, cols]
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=A.shape[0]))]
    return len(rows), None, cols, ptr, A[rows, cols]

def find_wcp(A_type, A, c_l, c_u, x_l, x_u, **g):
    m, n = A.shape
    A_ne, A_row, A_col, A_ptr, A_val = a_storage(A, A_type)
    options = wcp.initialize()
    options['print_level'] = 0
    # use the LAPACK solvers sytr and potr rather than the defaults, which
    # need the environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    options['fdc_options']['symmetric_linear_solver'] = 'sytr'
    options['sbls_options']['symmetric_linear_solver'] = 'sytr'
    options['sbls_options']['definite_linear_solver'] = 'potr'
    wcp.load(n, m, A_type, A_ne, A_row, A_col, A_ptr, options)
    x, c, y_l, y_u, z_l, z_u, x_stat, c_stat \
        = wcp.find_wcp(n, m, A_ne, A_val, c_l, c_u, x_l, x_u, np.zeros(n),
                       np.zeros(m), np.zeros(m), np.zeros(n), np.zeros(n),
                       **g)
    inform = wcp.information()
    wcp.terminate()
    return x, c, y_l, y_u, z_l, z_u, x_stat, c_stat, inform

@pytest.fixture(params=[(problem, A_type) for problem in PROBLEMS
                        for A_type in A_STORAGE],
                ids=lambda p: f'{p[0]},A={p[1]}')
def solution(request):
    problem, A_type = request.param
    p = PROBLEMS[problem]
    return problem, find_wcp(A_type, p['A'], p['c_l'], p['c_u'], p['x_l'],
                             p['x_u'], g=p['g'])


def test_wcp_status(solution):
    problem, (x, c, y_l, y_u, z_l, z_u, x_stat, c_stat, inform) = solution
    assert inform['status'] == 0


def test_wcp_primal(solution):
    # c = A x, x lies strictly between its bounds, and so does c, apart
    # from the equality constraint
    problem, (x, c, y_l, y_u, z_l, z_u, x_stat, c_stat, inform) = solution
    p = PROBLEMS[problem]
    c_l, c_u, x_l, x_u = p['c_l'], p['c_u'], p['x_l'], p['x_u']
    assert c == pytest.approx(p['A'] @ x, abs=1e-8)
    assert np.all(x > x_l) and np.all(x < x_u)
    equality = c_l == c_u
    assert c[equality] == pytest.approx(c_l[equality], abs=1e-8)
    assert np.all(c[~equality] > c_l[~equality])
    assert np.all(c[~equality] < c_u[~equality])


def test_wcp_complementarity(solution):
    # each dual for a finite bound (other than those of the equality
    # constraint) has the correct strict sign, and its product with the
    # distance to the bound is a positive target, no larger than the
    # largest final target (divided by the acceptance fraction, 1 by
    # default); duals for infinite bounds are zero
    problem, (x, c, y_l, y_u, z_l, z_u, x_stat, c_stat, inform) = solution
    p = PROBLEMS[problem]
    c_l, c_u, x_l, x_u = p['c_l'], p['c_u'], p['x_l'], p['x_u']
    mu_max = inform['mu_final_target_max'] * (1.0 + 1e-8)
    inequality = c_l < c_u
    for v, v_l, v_u, d_l, d_u in [(x, x_l, x_u, z_l, z_u),
                                  (c[inequality], c_l[inequality],
                                   c_u[inequality], y_l[inequality],
                                   y_u[inequality])]:
        lower, upper = np.isfinite(v_l), np.isfinite(v_u)
        assert np.all(d_l[lower] > 0.0) and np.all(d_u[upper] < 0.0)
        assert np.all(d_l[~lower] == 0.0) and np.all(d_u[~upper] == 0.0)
        mu_l = (v - v_l)[lower] * d_l[lower]
        mu_u = (v - v_u)[upper] * d_u[upper]
        assert np.all(mu_l > 0.0) and np.all(mu_l <= mu_max)
        assert np.all(mu_u > 0.0) and np.all(mu_u <= mu_max)


def test_wcp_dual(request, solution):
    # g = A^T ( y_l + y_u ) + z_l + z_u
    problem, (x, c, y_l, y_u, z_l, z_u, x_stat, c_stat, inform) = solution
    p = PROBLEMS[problem]
    if np.any(p['c_l'] == p['c_u']):
        request.applymarker(EQUALITY_XFAIL)
    assert p['g'] == pytest.approx(p['A'].T @ (y_l + y_u) + z_l + z_u,
                                   abs=1e-8)


def test_wcp_feasible(request, solution):
    # the point is strictly feasible unless there is no strict interior
    problem, (x, c, y_l, y_u, z_l, z_u, x_stat, c_stat, inform) = solution
    if problem == 'free':
        request.applymarker(FREE_XFAIL)
    assert inform['feasible'] == PROBLEMS[problem]['feasible']


def test_wcp_status_arrays(request, solution):
    # x_stat and c_stat are 0 for variables and constraints that lie
    # between their bounds, and 3 for constraints whose bounds are equal
    problem, (x, c, y_l, y_u, z_l, z_u, x_stat, c_stat, inform) = solution
    p = PROBLEMS[problem]
    if np.any(np.isinf(p['x_l']) & np.isinf(p['x_u'])):
        request.applymarker(FREE_XFAIL)
    assert np.all(x_stat == 0)
    assert np.all(c_stat == np.where(p['c_l'] == p['c_u'], 3, 0))


@pytest.mark.parametrize('A_type', A_STORAGE)
def test_wcp_box(A_type):
    # the point is the centre of the box, with z_l = - z_u
    p = PROBLEMS['box']
    x, c, y_l, y_u, z_l, z_u, x_stat, c_stat, inform \
        = find_wcp(A_type, p['A'], p['c_l'], p['c_u'], p['x_l'], p['x_u'],
                   g=p['g'])
    assert inform['status'] == 0
    assert x == pytest.approx(x_star_box, abs=1e-8)
    assert z_l == pytest.approx(-z_u, abs=1e-8)


@pytest.mark.parametrize('A_type', A_STORAGE)
def test_wcp_g_default(A_type):
    # omitting g, or passing g = None, is the same as passing g = 0
    p = PROBLEMS['interior']
    args = (A_type, p['A'], p['c_l'], p['c_u'], p['x_l'], p['x_u'])
    x, c, y_l, y_u, z_l, z_u, x_stat, c_stat, inform \
        = find_wcp(*args, g=np.zeros(3))
    for g in [{}, {'g': None}]:
        x_g, c_g, y_l_g, y_u_g, z_l_g, z_u_g, x_stat_g, c_stat_g, inform_g \
            = find_wcp(*args, **g)
        assert inform_g['status'] == 0
        assert x_g == pytest.approx(x, abs=1e-12)
        assert y_l_g + y_u_g == pytest.approx(y_l + y_u, abs=1e-12)
        assert z_l_g + z_u_g == pytest.approx(z_l + z_u, abs=1e-12)


@pytest.mark.xfail(strict=True,
                   reason='wcp flags infeasible problems as -17, not -5 '
                   '(see https://github.com/ralna/GALAHAD/pull/615)')
def test_wcp_infeasible():
    # 0 <= x_0, x_1 <= 1 and x_0 + x_1 >= 3 are inconsistent
    with pytest.warns(RuntimeWarning):
        x, c, y_l, y_u, z_l, z_u, x_stat, c_stat, inform \
            = find_wcp('coordinate', np.array([[1.0, 1.0]]), np.array([3.0]),
                       np.array([infinity]), np.zeros(2), np.ones(2),
                       g=np.zeros(2))
    assert inform['status'] == -5
