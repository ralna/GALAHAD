from galahad import l2rt
import numpy as np
import pytest


# l2rt finds an (approximate) global minimizer of the regularized
# Euclidean-norm objective r(x) = nu(x) + sigma / p ||x||^p, where
# nu(x) = sqrt( ||A x - b||^2 + mu ||x||^2 ), with mu >= 0, sigma >= 0 and
# p >= 2, by a Krylov method that only needs the sums u + A v and
# v + A^T u, which the caller forms by reverse communication. x solves the
# problem if ( A^T A + lambda I ) x = A^T b with
# lambda = mu + sigma ||x||^( p - 2 ) nu(x). inform reports lambda
# ('multiplier'), r(x) ('obj'), ||x||, ||A x - b|| ('r_norm') and
# ||A^T ( A x - b ) + lambda x|| ('Atr_norm').

# 1) as in test_l2rt.py: n = 50, A = ( I ; diag( 1, ..., n ) ),
# b = ( 1, ..., 1 ), sigma = 1, and (mu, p) = (1, 3) as there, (0, 3) and
# (1, 2). The problem is separable, x_i = ( 1 + i ) / ( 1 + i^2 + lambda ),
# but lambda has no closed form, so check the conditions above
n_e = 50
A_e = np.vstack([np.eye(n_e), np.diag(np.arange(1.0, n_e + 1))])
b_e = np.ones(2 * n_e)

# 2) a small problem built from a chosen x and lambda: A = ( I ; diag( 1, 2 ) )
# so that A^T A = diag( 2, 5 ), and b = ( 9/10, 4/5, 9/10, 2 ), so that
# A^T b = ( 9/5, 24/5 ). x = ( 3, 4 ) / 5, with ||x|| = 1 and
# A x - b = - ( 3, 0, 3, 4 ) / 10, satisfies ( A^T A + I ) x = A^T b, so
# lambda = mu + sigma nu(x) must be 1, for any p. With mu = 0, this is
# sigma = 10 / sqrt( 34 ), when nu = sqrt( 34 ) / 10; with mu = 3/20,
# nu = sqrt( 34/100 + 15/100 ) = 7/10 and sigma = 17/14. Then
# r = nu + sigma / p.
# With sigma = mu = 0, x = ( A^T A )^-1 A^T b = ( 9/10, 24/25 ) and
# r = ||A x - b|| = 2 sqrt( 5 ) / 25
A = np.vstack([np.eye(2), np.diag([1.0, 2.0])])
b = np.array([0.9, 0.8, 0.9, 2.0])
x_star = np.array([0.6, 0.8])
SHIFTS = {'mu=0': (0.0, 10.0 / np.sqrt(34.0), np.sqrt(34.0) / 10.0),
          'mu=3/20': (0.15, 17.0 / 14.0, 0.7)}

# for p > 2, l2rt ignores the shift
SHIFT_XFAIL = pytest.mark.xfail(
    strict=True, reason='l2rt ignores the shift mu when p > 2 (see '
    'https://github.com/ralna/GALAHAD/issues/641)')
# l2rt reports the last component of the subspace solution rather than the
# multiplier
MULTIPLIER_XFAIL = pytest.mark.xfail(
    strict=True, reason='l2rt reports the wrong multiplier (see '
    'https://github.com/ralna/GALAHAD/pull/640)')

def options(**opts):
    # stop only once the gradient is tiny
    o = l2rt.initialize()
    o['print_level'] = 0
    o['stop_relative'] = 1e-14
    o['stop_absolute'] = 0.0
    o.update(opts)
    return o

def solve(power, weight, shift, A, b, **opts):
    # solve the problem by reverse communication, returning the final
    # status, x and inform
    m, n = A.shape
    l2rt.load_options(options(**opts))
    status = 1
    u = b.copy()
    v = np.zeros(n)
    while True:
        status, x, u, v = l2rt.solve_problem(status, m, n, power, weight,
                                             shift, u, v)
        if status == 2:  # u <- u + A v
            u = u + A @ v
        elif status == 3:  # v <- v + A^T u
            v = v + A.T @ u
        elif status == 4:  # restart with u = b
            u = b.copy()
        else:
            break
    return status, x, l2rt.information()

def check_solution(x, inform, power, weight, shift, A, b):
    # check the conditions above, and the reported values
    norm_x = np.linalg.norm(x)
    r = A @ x - b
    nu = np.sqrt(r @ r + shift * norm_x ** 2)
    lam = shift + weight * norm_x ** (power - 2.0) * nu
    assert A.T @ r + lam * x == pytest.approx(np.zeros(len(x)), abs=1e-8)
    assert inform['x_norm'] == pytest.approx(norm_x, abs=1e-8)
    assert inform['r_norm'] == pytest.approx(np.linalg.norm(r), abs=1e-8)
    assert inform['obj'] == pytest.approx(
        nu + weight / power * norm_x ** power, abs=1e-8)

def example_multiplier(x):
    # the first component of ( A^T A + lambda I ) x = A^T b for problem 1)
    return 2.0 / x[0] - 2.0


@pytest.fixture
def terminate():
    yield
    l2rt.terminate()


@pytest.mark.parametrize('shift, power', [
    pytest.param(1.0, 3.0, marks=SHIFT_XFAIL), (0.0, 3.0), (1.0, 2.0)],
    ids=['mu=1,p=3', 'mu=0,p=3', 'mu=1,p=2'])
def test_l2rt_example(terminate, shift, power):
    # problem 1)
    status, x, inform = solve(power, 1.0, shift, A_e, b_e)
    assert status == 0
    assert inform['status'] == 0
    check_solution(x, inform, power, 1.0, shift, A_e, b_e)


@MULTIPLIER_XFAIL
def test_l2rt_example_multiplier(terminate):
    # problem 1) with mu = 0: the reported multiplier is the one x satisfies
    status, x, inform = solve(3.0, 1.0, 0.0, A_e, b_e)
    assert inform['multiplier'] == pytest.approx(example_multiplier(x),
                                                 rel=1e-8)


@pytest.mark.parametrize('power', [2.0, 3.0], ids=['p=2', 'p=3'])
@pytest.mark.parametrize('case', SHIFTS)
def test_l2rt(request, terminate, case, power):
    # problem 2)
    shift, weight, nu = SHIFTS[case]
    if shift > 0.0 and power > 2.0:
        request.applymarker(SHIFT_XFAIL)
    status, x, inform = solve(power, weight, shift, A, b)
    assert status == 0
    assert x == pytest.approx(x_star, abs=1e-8)
    assert inform['obj'] == pytest.approx(nu + weight / power, abs=1e-8)
    assert inform['r_norm'] == pytest.approx(np.sqrt(34.0) / 10.0,
                                             abs=1e-8)
    check_solution(x, inform, power, weight, shift, A, b)


@MULTIPLIER_XFAIL
@pytest.mark.parametrize('power', [2.0, 3.0], ids=['p=2', 'p=3'])
def test_l2rt_multiplier(terminate, power):
    # problem 2) with mu = 0: the reported multiplier
    shift, weight, nu = SHIFTS['mu=0']
    status, x, inform = solve(power, weight, shift, A, b)
    assert inform['multiplier'] == pytest.approx(1.0, abs=1e-8)


def test_l2rt_least_squares(terminate):
    # problem 2) with sigma = mu = 0 (solved in a single pass)
    status, x, inform = solve(3.0, 0.0, 0.0, A, b)
    assert status == 0
    assert x == pytest.approx(np.array([0.9, 0.96]), abs=1e-8)
    assert inform['obj'] == pytest.approx(2.0 * np.sqrt(5.0) / 25.0,
                                          abs=1e-8)
    check_solution(x, inform, 3.0, 0.0, 0.0, A, b)


@pytest.mark.parametrize('power, weight, shift', [
    (3.0, -1.0, 0.0), (3.0, 1.0, -1.0), (1.5, 1.0, 0.0)],
    ids=['sigma<0', 'mu<0', 'p<2'])
def test_l2rt_bad_parameters(terminate, power, weight, shift):
    # sigma and mu must be non-negative and p >= 2 (status -3)
    with pytest.raises(ValueError):
        solve(power, weight, shift, A, b)
    assert l2rt.information()['status'] == -3


def test_l2rt_iteration_limit(terminate):
    # the iteration limit stops l2rt successfully, with the minimizer of r
    # within the Krylov subspace generated so far
    status, x, inform = solve(3.0, 1.0, 0.0, A_e, b_e)
    obj = inform['obj']
    l2rt.terminate()
    status, x, inform = solve(3.0, 1.0, 0.0, A_e, b_e, itmax=1)
    assert status == 0
    assert inform['iter'] == 1
    r = A_e @ x - b_e
    assert inform['obj'] == pytest.approx(
        np.linalg.norm(r) + np.linalg.norm(x) ** 3 / 3.0, abs=1e-8)
    assert inform['obj'] > obj


# a single pass (sigma = mu = 0) stopped by the iteration limit crashes
# Python, so it cannot be a strict xfail
@pytest.mark.skip(reason='l2rt crashes when a single pass reaches the '
                  'iteration limit (see '
                  'https://github.com/ralna/GALAHAD/pull/640)')
def test_l2rt_iteration_limit_single_pass(terminate):
    status, x, inform = solve(3.0, 0.0, 0.0, A_e, b_e, itmax=1)
    assert status == 0
    assert inform['iter'] == 1
