from galahad import dgo
import numpy as np
import pytest


# test problem:
# minimize (x_0 + x_2 + p)^2 + (x_1 + x_2)^2 + mag cos(freq x_0) + x_0 + x_1 + x_2
# over [x_l,x_u]
p = 4
freq = 10
mag = 1000
n = 3
x_l = np.array([-10.0, -10.0, -10.0])
x_u = np.array([1.0, 1.0, 1.0])

# Hessian sparsity (lower triangle, coordinate format)
H_type = 'coordinate'
H_ne = 5
H_row = np.array([0, 2, 1, 2, 2])
H_col = np.array([0, 0, 1, 1, 2])
H_ptr = None

def eval_f(x):
    return (x[0] + x[2] + p)**2 + (x[1] + x[2])**2 \
        + mag * np.cos(freq * x[0]) + x[0] + x[1] + x[2]

def eval_g(x):
    return np.array([2. * (x[0] + x[2] + p)
                     - mag * freq * np.sin(freq * x[0]) + 1.,
                     2. * (x[1] + x[2]) + 1.,
                     2. * (x[0] + x[2] + p) + 2. * (x[1] + x[2]) + 1.])

def eval_h(x):
    return np.array([2. - mag * freq * freq * np.cos(freq * x[0]),
                     2., 2., 2., 4.])

# dgo does not terminate on this problem: it always stops with status -99
# (budget exhausted) when maxit is reached. With maxit = 5000 the best point
# found is the interior local minimizer near x_0 = -3 pi / 2. Setting g = 0
# gives x_1 + x_2 = -1/2, x_0 + x_2 + p = 0 and mag freq sin(freq x_0) = 1,
# so in closed form:
x0_star = -1.5 * np.pi - np.arcsin(1.0 / (mag * freq)) / freq
x_star = np.array([x0_star, x0_star + p - 0.5, -p - x0_star])
f_star = mag * np.cos(freq * x0_star) + x0_star - 0.25
# (this is not the global minimizer, which lies at x ~ [-5.3407, -1.5, 1]
# with x_2 on its upper bound, f ~ -1005.4746; dgo does not find it even
# with much larger maxit and max_evals)

@pytest.fixture
def solution():
    options = dgo.initialize()
    options['maxit'] = 5000
    dgo.load(n, x_l, x_u, H_type, H_ne, H_row, H_col, H_ptr, options=options)
    x0 = np.array([1., 1., 1.])
    x, g = dgo.solve(n, H_ne, x0, eval_f, eval_g, eval_h)
    inform = dgo.information()
    dgo.terminate()
    return x, g, inform


def test_dgo_status(solution):
    x, g, inform = solution
    # -99: the iteration budget was exhausted
    assert inform['status'] == -99


def test_dgo_solution(solution):
    x, g, inform = solution
    assert np.all(x_l <= x) and np.all(x <= x_u)
    # returned values are consistent with the objective at x
    assert g == pytest.approx(eval_g(x), abs=1e-8)
    assert inform['obj'] == pytest.approx(eval_f(x))
    # the minimizer is an interior stationary point with positive
    # definite Hessian
    assert g == pytest.approx(np.zeros(n), abs=1e-5)
    H = np.zeros((n, n))
    H[H_row, H_col] = eval_h(x)
    H = H + np.tril(H, -1).T
    assert np.all(np.linalg.eigvalsh(H) > 0.0)
    assert x == pytest.approx(x_star, abs=1e-5)
    assert inform['obj'] == pytest.approx(f_star, abs=1e-8)
