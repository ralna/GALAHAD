from galahad import gltr
import numpy as np
import pytest


# gltr finds an (approximate) global minimizer of
# q(x) = g^T x + 1/2 x^T H x within the trust region ||x||_M <= radius,
# where ||x||_M = sqrt( x^T M x ), by a Lanczos method that only needs
# products with H and M^-1, which the caller forms by reverse
# communication. x solves the problem if, for some lambda >= 0,
# ( H + lambda M ) x = - g, lambda ( ||x||_M - radius ) = 0,
# ||x||_M <= radius and H + lambda M is positive semi-definite. The
# Lanczos method is exact after at most n iterations.

# 1) as in test_gltr.py: n = 100, H = tridiag( 1, 2, 1 ), M = 2 I and
# g = ( 1, ..., 1 ), with radius 1 and then (by a restart) 0.1. H is
# positive definite and ||H^-1 g||_M > 1, so the solutions lie on the
# boundary; there is no closed form, so check the conditions above
n_e = 100
H_e = 2.0 * np.eye(n_e) + np.eye(n_e, k=1) + np.eye(n_e, k=-1)
M_e = 2.0 * np.eye(n_e)
g_e = np.ones(n_e)

# well-posed small problems, built from a chosen x and lambda with
# H + lambda M positive definite by setting g = - ( H + lambda M ) x:
#     ( 1 0 4 )
# H = ( 0 2 0 ), which is indefinite, and M = diag( 1, 2, 1 ):
#     ( 4 0 3 )
# 2) M = I, x = ( 2, -1, -2 ) / 3 with ||x|| = 1 and lambda = 3, so
# g = ( 0, 5, 4 ) / 3 and q = - 13/9 - 7/9 = - 20/9
# 3) M as above, x = ( 1, 1, 1 ) / 2 with ||x||_M = 1 and lambda = 3, so
# g = - ( 4, 4, 5 ) and q = - 13/2 + 7/4 = - 19/4
# 4) the positive definite H + 3 I with g as in 2) and radius 2: the
# unconstrained minimizer x = ( 2, -1, -2 ) / 3 lies inside the region, so
# lambda = 0 and q = - 13/18
# 5) equality_problem, requiring ||x|| = 2, with H + 3 I,
# x = ( 4, -2, -4 ) / 3 and lambda = - 1/2 (H + 5/2 I is positive
# definite), so g = - ( H + 5/2 I ) x = ( 2, 9, 6 ) / 3 and q = - 8/9
H = np.array([[1.0, 0.0, 4.0], [0.0, 2.0, 0.0], [4.0, 0.0, 3.0]])
M = np.diag([1.0, 2.0, 1.0])
H_pd = H + 3.0 * np.eye(3)
PROBLEMS = {
    'unit': dict(H=H, M=None, g=np.array([0.0, 5.0, 4.0]) / 3.0, radius=1.0,
                 x=np.array([2.0, -1.0, -2.0]) / 3.0, lam=3.0, q=-20.0 / 9.0),
    'M': dict(H=H, M=M, g=-np.array([4.0, 4.0, 5.0]), radius=1.0,
              x=np.array([0.5, 0.5, 0.5]), lam=3.0, q=-19.0 / 4.0),
    'interior': dict(H=H_pd, M=None, g=np.array([0.0, 5.0, 4.0]) / 3.0,
                     radius=2.0, x=np.array([2.0, -1.0, -2.0]) / 3.0,
                     lam=0.0, q=-13.0 / 18.0),
    'equality': dict(H=H_pd, M=None, g=np.array([2.0, 9.0, 6.0]) / 3.0,
                     radius=2.0, x=np.array([4.0, -2.0, -4.0]) / 3.0,
                     lam=-0.5, q=-8.0 / 9.0, equality_problem=True),
}

def options(M=None, **opts):
    # options for a problem whose norm is given by M (None for M = I),
    # stopping only once the gradient is tiny
    o = gltr.initialize()
    o['print_level'] = 0
    o['unitm'] = M is None
    o['stop_relative'] = 1e-14
    o['stop_absolute'] = 0.0
    o.update(opts)
    return o

def iterate(status, radius, g, H, M):
    # run the reverse-communication loop from the given status, returning
    # the final status and x
    n = len(g)
    r = g.copy()
    v = np.zeros(n)
    while True:
        status, x, r, v = gltr.solve_problem(status, n, radius, r, v)
        if status == 2:  # apply M^-1
            v = np.linalg.solve(M, v)
        elif status == 3:  # form H v
            v = H @ v
        elif status == 5:  # restart with r = g
            r = g.copy()
        else:
            return status, x

def solve(g, radius, H, M=None, **opts):
    # solve the problem, returning the final status, x and inform
    gltr.load_options(options(M, **opts))
    status, x = iterate(1, radius, g, H, M)
    inform = gltr.information()
    return status, x, inform

def check_solution(x, inform, g, radius, H, M=None, lam=None):
    # check the conditions above for the reported multiplier, and the
    # reported ||x||_M and q
    M = np.eye(len(g)) if M is None else M
    if lam is None:
        lam = inform['multiplier']
        assert lam >= 0.0
    assert inform['multiplier'] == pytest.approx(lam, abs=1e-8)
    norm_x = np.sqrt(x @ M @ x)
    assert norm_x <= radius + 1e-8
    if lam != 0.0:
        assert norm_x == pytest.approx(radius, abs=1e-8)
    assert (H + lam * M) @ x + g == pytest.approx(np.zeros(len(g)), abs=1e-8)
    assert inform['mnormx'] == pytest.approx(norm_x, abs=1e-8)
    assert inform['obj'] == pytest.approx(g @ x + 0.5 * x @ H @ x, abs=1e-8)


@pytest.fixture
def terminate():
    yield
    gltr.terminate()


@pytest.mark.parametrize('radius', [1.0, 0.1])
def test_gltr_example(terminate, radius):
    # problem 1)
    status, x, inform = solve(g_e, radius, H_e, M_e)
    assert status == 0
    assert inform['status'] == 0
    assert inform['multiplier'] > 0.0
    check_solution(x, inform, g_e, radius, H_e, M_e)


def test_gltr_example_restart(terminate):
    # problem 1), with radius 0.1 found by a restart (status 4) after
    # solving with radius 1
    gltr.load_options(options(M_e))
    status, x = iterate(1, 1.0, g_e, H_e, M_e)
    assert status == 0
    status, x = iterate(4, 0.1, g_e, H_e, M_e)
    inform = gltr.information()
    assert status == 0
    check_solution(x, inform, g_e, 0.1, H_e, M_e)


@pytest.mark.parametrize('problem', PROBLEMS)
def test_gltr(terminate, problem):
    # problems 2) to 5)
    p = dict(PROBLEMS[problem])
    g, radius, Hp, Mp = p.pop('g'), p.pop('radius'), p.pop('H'), p.pop('M')
    x_star, lam, q = p.pop('x'), p.pop('lam'), p.pop('q')
    status, x, inform = solve(g, radius, Hp, Mp, **p)
    assert status == 0
    assert x == pytest.approx(x_star, abs=1e-8)
    assert inform['obj'] == pytest.approx(q, abs=1e-8)
    check_solution(x, inform, g, radius, Hp, Mp, lam)


def test_gltr_restart(terminate):
    # problem 2), with radius 1 found by a restart (status 4) after solving
    # with radius 2
    p = PROBLEMS['unit']
    gltr.load_options(options())
    status, x = iterate(1, 2.0, p['g'], p['H'], None)
    assert status == 0
    status, x = iterate(4, 1.0, p['g'], p['H'], None)
    inform = gltr.information()
    assert status == 0
    assert x == pytest.approx(p['x'], abs=1e-8)
    check_solution(x, inform, p['g'], 1.0, p['H'], None, p['lam'])


def test_gltr_radius_not_positive(terminate):
    # the radius must be positive (status -3)
    p = PROBLEMS['unit']
    with pytest.raises(ValueError):
        solve(p['g'], 0.0, p['H'])
    assert gltr.information()['status'] == -3


def test_gltr_iteration_limit_interior(terminate):
    # with radius 100 the solution of problem 1) is interior, and needs
    # more than one iteration (status -18)
    with pytest.warns(RuntimeWarning):
        status, x, inform = solve(g_e, 100.0, H_e, M_e, itmax=1)
    assert status == -18
    assert inform['status'] == -18


def test_gltr_iteration_limit_boundary(terminate):
    # once the boundary has been reached, the iteration limit instead stops
    # gltr successfully, with the best point on the boundary within the
    # Krylov subspace generated so far
    status, x, inform = solve(g_e, 1.0, H_e, M_e)
    obj = inform['obj']
    gltr.terminate()
    status, x, inform = solve(g_e, 1.0, H_e, M_e, itmax=1)
    assert status == 0
    assert inform['iter'] == 1
    assert np.sqrt(x @ M_e @ x) == pytest.approx(1.0, abs=1e-8)
    assert inform['obj'] == pytest.approx(g_e @ x + 0.5 * x @ H_e @ x,
                                          abs=1e-8)
    assert inform['obj'] > obj


def test_gltr_indefinite_m(terminate):
    # M must be positive definite (status -15)
    p = PROBLEMS['unit']
    with pytest.warns(RuntimeWarning):
        status, x, inform = solve(p['g'], 1.0, p['H'],
                                  np.diag([1.0, -1.0, 1.0]))
    assert status == -15


def test_gltr_steihaug_toint(terminate):
    # in Steihaug-Toint mode, gltr stops when the conjugate-gradient path
    # meets the boundary (status -30)
    status, x, inform = solve(g_e, 1.0, H_e, M_e, steihaug_toint=True)
    assert status == -30
    assert np.sqrt(x @ M_e @ x) == pytest.approx(1.0, abs=1e-8)


def test_gltr_f_min(terminate):
    # problem 2) with q( x ) = - 20/9 < f_min = -1 (status -44)
    p = PROBLEMS['unit']
    status, x, inform = solve(p['g'], 1.0, p['H'], f_min=-1.0)
    assert status == -44
    assert inform['obj'] < -1.0
