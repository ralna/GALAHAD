from galahad import lsrt
import numpy as np
import pytest


# lsrt finds an (approximate) global minimizer of the regularized
# least-squares objective r(x) = 1/2 ||A x - b||^2 + sigma / p ||x||^p, with
# sigma >= 0 and p >= 2, by a Krylov method that only needs the sums
# u + A v and v + A^T u, which the caller forms by reverse communication.
# x solves the problem if ( A^T A + lambda I ) x = A^T b with
# lambda = sigma ||x||^( p - 2 ). inform reports lambda ('multiplier'),
# r(x) ('obj'), ||x||, ||A x - b|| ('r_norm') and
# ||A^T ( A x - b ) + lambda x|| ('Atr_norm').

# 1) as in test_lsrt.py: n = 50, A = ( I ; diag( 1, ..., n ) ),
# b = ( 1, ..., 1 ), p = 3 and sigma = 1. The problem is separable,
# x_i = ( 1 + i ) / ( 1 + i^2 + lambda ), but lambda has no closed form, so
# check the conditions above
n_e = 50
A_e = np.vstack([np.eye(n_e), np.diag(np.arange(1.0, n_e + 1))])
b_e = np.ones(2 * n_e)

# 2) a small problem built from a chosen x and lambda: A = ( I ; diag( 1, 2 ) )
# so that A^T A = diag( 2, 5 ), and b = ( 9/10, 4/5, 9/10, 2 ), so that
# A^T b = ( 9/5, 24/5 ). x = ( 3, 4 ) / 5 has ||x|| = 1, so with sigma = 1,
# lambda = sigma ||x||^( p - 2 ) = 1 for any p, and ( A^T A + I ) x = A^T b.
# Then A x - b = - ( 3, 0, 3, 4 ) / 10, so r = 17/100 + 1/p. With sigma = 0,
# x = ( A^T A )^-1 A^T b = ( 9/10, 24/25 ), lambda = 0 and r = 2/125
A = np.vstack([np.eye(2), np.diag([1.0, 2.0])])
b = np.array([0.9, 0.8, 0.9, 2.0])
x_star = np.array([0.6, 0.8])
x_star_0 = np.array([0.9, 0.96])


def options(**opts):
    # stop only once the gradient is tiny
    o = lsrt.initialize()
    o['print_level'] = 0
    o['stop_relative'] = 1e-14
    o['stop_absolute'] = 0.0
    o.update(opts)
    return o

def solve(power, weight, A, b, **opts):
    # solve the problem by reverse communication, returning the final
    # status, x and inform
    m, n = A.shape
    lsrt.load_options(options(**opts))
    status = 1
    u = b.copy()
    v = np.zeros(n)
    while True:
        status, x, u, v = lsrt.solve_problem(status, m, n, power, weight, u, v)
        if status == 2:  # u <- u + A v
            u = u + A @ v
        elif status == 3:  # v <- v + A^T u
            v = v + A.T @ u
        elif status == 4:  # restart with u = b
            u = b.copy()
        else:
            break
    return status, x, lsrt.information()

def check_solution(x, inform, power, weight, A, b, lam):
    # check the conditions above for the multiplier lam, and the reported
    # values
    norm_x = np.linalg.norm(x)
    assert lam == pytest.approx(weight * norm_x ** (power - 2.0), abs=1e-8)
    r = A @ x - b
    assert A.T @ r + lam * x == pytest.approx(np.zeros(len(x)), abs=1e-8)
    assert inform['x_norm'] == pytest.approx(norm_x, abs=1e-8)
    assert inform['r_norm'] == pytest.approx(np.linalg.norm(r), abs=1e-8)
    assert inform['obj'] == pytest.approx(
        0.5 * r @ r + weight / power * norm_x ** power, abs=1e-8)

def example_multiplier(x):
    # the first component of ( A^T A + lambda I ) x = A^T b for problem 1)
    return 2.0 / x[0] - 2.0


@pytest.fixture
def terminate():
    yield
    lsrt.terminate()


@pytest.mark.parametrize('power', [2.0, 3.0], ids=['p=2', 'p=3'])
def test_lsrt_example(terminate, power):
    # problem 1), and its variant with p = 2 (solved in a single pass)
    status, x, inform = solve(power, 1.0, A_e, b_e)
    assert status == 0
    assert inform['status'] == 0
    check_solution(x, inform, power, 1.0, A_e, b_e, example_multiplier(x))


@pytest.mark.parametrize('power', [2.0, 3.0], ids=['p=2', 'p=3'])
def test_lsrt_example_multiplier(terminate, power):
    # problem 1): the reported multiplier is the one x satisfies
    status, x, inform = solve(power, 1.0, A_e, b_e)
    assert inform['multiplier'] == pytest.approx(example_multiplier(x),
                                                 rel=1e-8)


@pytest.mark.parametrize('power', [2.0, 3.0, 4.0],
                         ids=['p=2', 'p=3', 'p=4'])
def test_lsrt(terminate, power):
    # problem 2) with sigma = 1
    status, x, inform = solve(power, 1.0, A, b)
    assert status == 0
    assert x == pytest.approx(x_star, abs=1e-8)
    assert inform['obj'] == pytest.approx(0.17 + 1.0 / power, abs=1e-8)
    assert inform['r_norm'] == pytest.approx(np.sqrt(34.0) / 10.0,
                                             abs=1e-8)
    check_solution(x, inform, power, 1.0, A, b, 1.0)


@pytest.mark.parametrize('power', [2.0, 3.0, 4.0],
                         ids=['p=2', 'p=3', 'p=4'])
def test_lsrt_multiplier(terminate, power):
    # problem 2) with sigma = 1: the reported multiplier
    status, x, inform = solve(power, 1.0, A, b)
    assert inform['multiplier'] == pytest.approx(1.0, abs=1e-8)


def test_lsrt_zero_weight(terminate):
    # problem 2) with sigma = 0: a linear least-squares problem
    status, x, inform = solve(3.0, 0.0, A, b)
    assert status == 0
    assert x == pytest.approx(x_star_0, abs=1e-8)
    assert inform['multiplier'] == pytest.approx(0.0, abs=1e-8)
    assert inform['obj'] == pytest.approx(2.0 / 125.0, abs=1e-8)
    check_solution(x, inform, 3.0, 0.0, A, b, 0.0)


@pytest.mark.parametrize('power, weight', [(3.0, -1.0), (1.5, 1.0)],
                         ids=['sigma<0', 'p<2'])
def test_lsrt_bad_parameters(terminate, power, weight):
    # sigma must be non-negative and p >= 2 (status -3)
    with pytest.raises(ValueError):
        solve(power, weight, A, b)
    assert lsrt.information()['status'] == -3


def test_lsrt_iteration_limit(terminate):
    # the iteration limit stops lsrt successfully, with the minimizer of r
    # within the Krylov subspace generated so far
    status, x, inform = solve(3.0, 1.0, A_e, b_e)
    obj = inform['obj']
    lsrt.terminate()
    status, x, inform = solve(3.0, 1.0, A_e, b_e, itmax=1)
    assert status == 0
    assert inform['iter'] == 1
    r = A_e @ x - b_e
    assert inform['obj'] == pytest.approx(
        0.5 * r @ r + np.linalg.norm(x) ** 3 / 3.0, abs=1e-8)
    assert inform['obj'] > obj


# a single pass (p = 2) stopped by the iteration limit
def test_lsrt_iteration_limit_single_pass(terminate):
    status, x, inform = solve(2.0, 1.0, A_e, b_e, itmax=1)
    assert status == 0
    assert inform['iter'] == 1
