from galahad import glrt
import numpy as np
import pytest


# glrt finds an (approximate) global minimizer of the regularized quadratic
# r(x) = q(x) + sigma / p ||x||_M^p, where q(x) = g^T x + 1/2 x^T H x,
# ||x||_M = sqrt( x^T M x ), p >= 2 and sigma > 0, by a Lanczos method that
# only needs products with H and M^-1, which the caller forms by reverse
# communication. x solves the problem if ( H + lambda M ) x = - g with
# lambda = sigma ||x||_M^( p - 2 ) and H + lambda M positive semi-definite.
# inform['obj'] reports q(x) and inform['obj_regularized'] r(x). The Lanczos
# method is exact after at most n iterations.

# 1) as in test_glrt.py: n = 100, H = tridiag( 1, 2, 1 ), M = 2 I,
# g = ( 1, ..., 1 ), p = 3, and sigma = 1 and then (by a restart) 10.
# There is no closed form, so check the conditions above
n_e = 100
H_e = 2.0 * np.eye(n_e) + np.eye(n_e, k=1) + np.eye(n_e, k=-1)
M_e = 2.0 * np.eye(n_e)
g_e = np.ones(n_e)

# well-posed small problems, built from a chosen x and lambda with
# H + lambda M positive definite by setting g = - ( H + lambda M ) x:
#     ( 1 0 4 )
# H = ( 0 2 0 ), which is indefinite, and M = diag( 1, 2, 1 ):
#     ( 4 0 3 )
# 2) M = I, x = ( 2, -1, -2 ) with ||x|| = 3 and lambda = 3, so
# g = ( 0, 5, 4 ) and q = - 13 - 7 = - 20. lambda = sigma ||x||^( p - 2 )
# = 3 for p = 2, 3 and 4 with sigma = 3, 1 and 1/3, when
# r = q + sigma / p 3^p = - 13/2, - 11 and - 53/4
# 3) M as above, p = 3, sigma = 1, x = ( 3, 3, 3 ) / 2 with ||x||_M = 3 and
# lambda = 3, so g = - ( 12, 12, 15 ), q = - 171/4 and r = q + 9 = - 135/4
H = np.array([[1.0, 0.0, 4.0], [0.0, 2.0, 0.0], [4.0, 0.0, 3.0]])
M = np.diag([1.0, 2.0, 1.0])
g_b = np.array([0.0, 5.0, 4.0])
x_star_b = np.array([2.0, -1.0, -2.0])
POWERS = [(2.0, 3.0, -13.0 / 2.0), (3.0, 1.0, -11.0),
          (4.0, 1.0 / 3.0, -53.0 / 4.0)]
g_m = -np.array([12.0, 12.0, 15.0])
x_star_m = np.array([1.5, 1.5, 1.5])

def options(M=None, **opts):
    # options for a problem whose norm is given by M (None for M = I),
    # stopping only once the gradient is tiny
    o = glrt.initialize()
    o['print_level'] = 0
    o['unitm'] = M is None
    o['stop_relative'] = 1e-14
    o['stop_absolute'] = 0.0
    o.update(opts)
    return o

def iterate(status, power, weight, g, H, M):
    # run the reverse-communication loop from the given status, returning
    # the final status and x
    n = len(g)
    r = g.copy()
    v = np.zeros(n)
    while True:
        status, x, r, v = glrt.solve_problem(status, n, power, weight, r, v)
        if status == 2:  # apply M^-1
            v = np.linalg.solve(M, v)
        elif status == 3:  # form H v
            v = H @ v
        elif status == 4:  # restart with r = g
            r = g.copy()
        else:
            return status, x

def solve(g, power, weight, H, M=None, **opts):
    # solve the problem, returning the final status, x and inform
    glrt.load_options(options(M, **opts))
    status, x = iterate(1, power, weight, g, H, M)
    inform = glrt.information()
    return status, x, inform

def check_solution(x, inform, g, power, weight, H, M=None):
    # check the conditions above for the reported multiplier, and the
    # reported ||x||_M, q and r
    M = np.eye(len(g)) if M is None else M
    lam = inform['multiplier']
    norm_x = np.sqrt(x @ M @ x)
    assert lam == pytest.approx(weight * norm_x ** (power - 2.0), abs=1e-8)
    assert (H + lam * M) @ x + g == pytest.approx(np.zeros(len(g)), abs=1e-8)
    assert np.linalg.eigvalsh(H + lam * M).min() >= -1e-8
    assert inform['xpo_norm'] == pytest.approx(norm_x, abs=1e-8)
    q = g @ x + 0.5 * x @ H @ x
    assert inform['obj'] == pytest.approx(q, abs=1e-8)
    assert inform['obj_regularized'] \
        == pytest.approx(q + weight / power * norm_x ** power, abs=1e-8)


@pytest.fixture
def terminate():
    yield
    glrt.terminate()


@pytest.mark.parametrize('weight', [1.0, 10.0])
def test_glrt_example(terminate, weight):
    # problem 1)
    status, x, inform = solve(g_e, 3.0, weight, H_e, M_e)
    assert status == 0
    assert inform['status'] == 0
    check_solution(x, inform, g_e, 3.0, weight, H_e, M_e)


def test_glrt_example_restart(terminate):
    # problem 1), with sigma = 10 found by a restart (status 6) after
    # solving with sigma = 1
    glrt.load_options(options(M_e))
    status, x = iterate(1, 3.0, 1.0, g_e, H_e, M_e)
    assert status == 0
    status, x = iterate(6, 3.0, 10.0, g_e, H_e, M_e)
    inform = glrt.information()
    assert status == 0
    check_solution(x, inform, g_e, 3.0, 10.0, H_e, M_e)


@pytest.mark.parametrize('power, weight, r_b', POWERS,
                         ids=['p=2', 'p=3', 'p=4'])
def test_glrt_powers(terminate, power, weight, r_b):
    # problem 2)
    status, x, inform = solve(g_b, power, weight, H)
    assert status == 0
    assert x == pytest.approx(x_star_b, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(3.0, abs=1e-8)
    assert inform['obj'] == pytest.approx(-20.0, abs=1e-8)
    assert inform['obj_regularized'] == pytest.approx(r_b, abs=1e-8)
    check_solution(x, inform, g_b, power, weight, H)


def test_glrt_m(terminate):
    # problem 3)
    status, x, inform = solve(g_m, 3.0, 1.0, H, M)
    assert status == 0
    assert x == pytest.approx(x_star_m, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(3.0, abs=1e-8)
    assert inform['obj'] == pytest.approx(-171.0 / 4.0, abs=1e-8)
    assert inform['obj_regularized'] == pytest.approx(-135.0 / 4.0,
                                                      abs=1e-8)
    check_solution(x, inform, g_m, 3.0, 1.0, H, M)


def test_glrt_restart(terminate):
    # problem 2) with p = 3, sigma = 1 found by a restart (status 6) after
    # solving with sigma = 1/2
    glrt.load_options(options())
    status, x = iterate(1, 3.0, 0.5, g_b, H, None)
    assert status == 0
    status, x = iterate(6, 3.0, 1.0, g_b, H, None)
    inform = glrt.information()
    assert status == 0
    assert x == pytest.approx(x_star_b, abs=1e-8)
    check_solution(x, inform, g_b, 3.0, 1.0, H)


@pytest.mark.parametrize('power, weight', [(3.0, -1.0), (1.5, 1.0)],
                         ids=['sigma<0', 'p<2'])
def test_glrt_bad_parameters(terminate, power, weight):
    # sigma must be non-negative and p >= 2 (status -3)
    with pytest.raises(ValueError):
        solve(g_b, power, weight, H)
    assert glrt.information()['status'] == -3


# 4) with sigma = 0, glrt minimizes q by the conjugate gradient method; with
# H = diag( 1, 2, 3 ) and g = ( 1, 1, 1 ), x = - ( 1, 1/2, 1/3 ) and
# q = - 11/12
def test_glrt_zero_weight(terminate):
    H_d = np.diag([1.0, 2.0, 3.0])
    g_d = np.ones(3)
    status, x, inform = solve(g_d, 3.0, 0.0, H_d)
    assert status == 0
    assert x == pytest.approx(-np.array([1.0, 0.5, 1.0 / 3.0]), abs=1e-8)
    assert inform['obj'] == pytest.approx(-11.0 / 12.0, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(0.0, abs=1e-8)


def test_glrt_unbounded(terminate):
    # with p = 2 and sigma = 1, H + I is indefinite, so r is unbounded
    # below (status -7)
    status, x, inform = solve(g_b, 2.0, 1.0, H)
    assert status == -7


def test_glrt_indefinite_m(terminate):
    # M must be positive definite (status -15)
    with pytest.warns(RuntimeWarning):
        status, x, inform = solve(g_b, 3.0, 1.0, H,
                                  np.diag([1.0, -1.0, 1.0]))
    assert status == -15


def test_glrt_iteration_limit(terminate):
    # the iteration limit stops glrt successfully, with the minimizer of r
    # within the Krylov subspace generated so far
    status, x, inform = solve(g_e, 3.0, 1.0, H_e, M_e)
    obj_regularized = inform['obj_regularized']
    glrt.terminate()
    status, x, inform = solve(g_e, 3.0, 1.0, H_e, M_e, itmax=1)
    assert status == 0
    assert inform['iter'] == 1
    norm_x = np.sqrt(x @ M_e @ x)
    q = g_e @ x + 0.5 * x @ H_e @ x
    assert inform['obj'] == pytest.approx(q, abs=1e-8)
    assert inform['obj_regularized'] \
        == pytest.approx(q + norm_x ** 3 / 3.0, abs=1e-8)
    assert inform['obj_regularized'] > obj_regularized
