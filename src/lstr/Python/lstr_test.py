from galahad import lstr
import numpy as np
import pytest


# lstr finds an (approximate) global minimizer of the least-squares
# objective 1/2 ||A x - b||^2 within the trust region ||x|| <= radius, by a
# Krylov method that only needs the sums u + A v and v + A^T u, which the
# caller forms by reverse communication. x solves the problem if, for some
# lambda >= 0, ( A^T A + lambda I ) x = A^T b, lambda ( ||x|| - radius ) = 0
# and ||x|| <= radius. inform reports lambda ('multiplier'), ||x||,
# ||A x - b|| ('r_norm') and ||A^T ( A x - b ) + lambda x|| ('Atr_norm').

# 1) as in test_lstr.py: n = 50, A = ( I ; diag( 1, ..., n ) ) and
# b = ( 1, ..., 1 ), with radius 1 and then (by a restart) 0.1. The problem
# is separable, x_i = ( 1 + i ) / ( 1 + i^2 + lambda ), but lambda has no
# closed form, so check the conditions above
n_e = 50
A_e = np.vstack([np.eye(n_e), np.diag(np.arange(1.0, n_e + 1))])
b_e = np.ones(2 * n_e)

# 2) a small problem built from a chosen x and lambda: A = ( I ; diag( 1, 2 ) )
# so that A^T A = diag( 2, 5 ), and b = ( 9/10, 4/5, 9/10, 2 ), so that
# A^T b = ( 9/5, 24/5 ). With radius 1, x = ( 3, 4 ) / 5 and lambda = 1 satisfy
# ( A^T A + lambda I ) x = A^T b with ||x|| = 1, and
# A x - b = - ( 3, 0, 3, 4 ) / 10, so ||A x - b|| = sqrt( 34 ) / 10. With
# radius 10, the unconstrained minimizer
# x = ( A^T A )^-1 A^T b = ( 9/10, 24/25 ) lies inside the region, so
# lambda = 0 and ||A x - b|| = 2 sqrt( 5 ) / 25
A = np.vstack([np.eye(2), np.diag([1.0, 2.0])])
b = np.array([0.9, 0.8, 0.9, 2.0])
PROBLEMS = {
    'boundary': dict(radius=1.0, x=np.array([0.6, 0.8]), lam=1.0,
                     r_norm=np.sqrt(34.0) / 10.0),
    'interior': dict(radius=10.0, x=np.array([0.9, 0.96]), lam=0.0,
                     r_norm=2.0 * np.sqrt(5.0) / 25.0),
}


def options(**opts):
    # stop only once the gradient is tiny
    o = lstr.initialize()
    o['print_level'] = 0
    o['stop_relative'] = 1e-14
    o['stop_absolute'] = 0.0
    o.update(opts)
    return o

def iterate(status, radius, A, b):
    # run the reverse-communication loop from the given status, returning
    # the final status and x
    m, n = A.shape
    u = b.copy()
    v = np.zeros(n)
    while True:
        status, x, u, v = lstr.solve_problem(status, m, n, radius, u, v)
        if status == 2:  # u <- u + A v
            u = u + A @ v
        elif status == 3:  # v <- v + A^T u
            v = v + A.T @ u
        elif status == 4:  # restart with u = b
            u = b.copy()
        else:
            return status, x

def solve(radius, A, b, **opts):
    # solve the problem, returning the final status, x and inform
    lstr.load_options(options(**opts))
    status, x = iterate(1, radius, A, b)
    inform = lstr.information()
    return status, x, inform

def check_solution(x, inform, radius, A, b, lam):
    # check the conditions above for the multiplier lam, and the reported
    # norms
    assert np.linalg.norm(x) <= radius + 1e-8
    if lam != 0.0:
        assert np.linalg.norm(x) == pytest.approx(radius, abs=1e-8)
    r = A @ x - b
    assert A.T @ r + lam * x == pytest.approx(np.zeros(len(x)), abs=1e-8)
    assert inform['x_norm'] == pytest.approx(np.linalg.norm(x), abs=1e-8)
    assert inform['r_norm'] == pytest.approx(np.linalg.norm(r), abs=1e-8)
    assert inform['Atr_norm'] == pytest.approx(0.0, abs=1e-8)

def example_multiplier(x):
    # the first component of ( A^T A + lambda I ) x = A^T b for problem 1)
    return 2.0 / x[0] - 2.0


@pytest.fixture
def terminate():
    yield
    lstr.terminate()


@pytest.mark.parametrize('radius', [1.0, 0.1])
def test_lstr_example(terminate, radius):
    # problem 1)
    status, x, inform = solve(radius, A_e, b_e)
    assert status == 0
    assert inform['status'] == 0
    lam = example_multiplier(x)
    assert lam > 0.0
    check_solution(x, inform, radius, A_e, b_e, lam)


def test_lstr_example_restart(terminate):
    # problem 1), with radius 0.1 found by a restart (status 5) after
    # solving with radius 1
    lstr.load_options(options())
    status, x = iterate(1, 1.0, A_e, b_e)
    assert status == 0
    status, x = iterate(5, 0.1, A_e, b_e)
    inform = lstr.information()
    assert status == 0
    check_solution(x, inform, 0.1, A_e, b_e, example_multiplier(x))


@pytest.mark.parametrize('radius', [1.0, 0.1])
def test_lstr_example_multiplier(terminate, radius):
    # problem 1): the reported multiplier is the one x satisfies
    status, x, inform = solve(radius, A_e, b_e)
    assert inform['multiplier'] == pytest.approx(example_multiplier(x),
                                                 rel=1e-8)


@pytest.mark.parametrize('problem', PROBLEMS)
def test_lstr(terminate, problem):
    # problem 2)
    p = PROBLEMS[problem]
    status, x, inform = solve(p['radius'], A, b)
    assert status == 0
    assert x == pytest.approx(p['x'], abs=1e-8)
    assert inform['r_norm'] == pytest.approx(p['r_norm'], abs=1e-8)
    check_solution(x, inform, p['radius'], A, b, p['lam'])


@pytest.mark.parametrize('problem', [
    'boundary', 'interior'])
def test_lstr_multiplier(terminate, problem):
    # problem 2): the reported multiplier
    p = PROBLEMS[problem]
    status, x, inform = solve(p['radius'], A, b)
    assert inform['multiplier'] == pytest.approx(p['lam'], abs=1e-8)


def test_lstr_restart(terminate):
    # problem 2) with radius 1, found by a restart (status 5) after solving
    # with radius 10
    p = PROBLEMS['boundary']
    lstr.load_options(options())
    status, x = iterate(1, 10.0, A, b)
    assert status == 0
    status, x = iterate(5, 1.0, A, b)
    inform = lstr.information()
    assert status == 0
    assert x == pytest.approx(p['x'], abs=1e-8)
    check_solution(x, inform, 1.0, A, b, p['lam'])


def test_lstr_radius_not_positive(terminate):
    # the radius must be positive (status -3)
    with pytest.raises(ValueError):
        solve(0.0, A, b)
    assert lstr.information()['status'] == -3


def test_lstr_iteration_limit(terminate):
    # problem 1) needs more than one iteration (status -18)
    with pytest.warns(RuntimeWarning):
        status, x, inform = solve(1.0, A_e, b_e, itmax=1)
    assert status == -18


def test_lstr_steihaug_toint(terminate):
    # in Steihaug-Toint mode, lstr stops when the iteration meets the
    # boundary (status -30)
    status, x, inform = solve(1.0, A_e, b_e, steihaug_toint=True)
    assert status == -30
    assert np.linalg.norm(x) == pytest.approx(1.0, abs=1e-8)
