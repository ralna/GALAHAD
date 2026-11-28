from galahad import bgo
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
x_u = np.array([0.5, 0.5, 0.5])

# Hessian sparsity (lower triangle, coordinate format)
H_type = 'coordinate'
H_ne = 5
H_row = np.array([0, 1, 2, 2, 2])
H_col = np.array([0, 1, 0, 1, 2])
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

# The global minimizer lies near x_0 = -3 pi / 2 with x_2 on its upper
# bound. Then g_1 = 0 gives x_1 = -1/2 - x_2 = -1, and x_0 solves
# 2 (x_0 + x_2 + p) + 1 - mag freq sin(freq x_0) = 0, computed by Newton's
# method. (With the default attempts_max = 10, bgo instead stops at the
# interior local minimizer near x_0 = -13 pi / 10, f ~ -1004.3341.)
x_star = np.array([-4.7123947324900435, -1.0, 0.5])
f_star = -1004.9172815557647

@pytest.fixture
def solution():
    options = bgo.initialize()
    options['attempts_max'] = 50
    bgo.load(n, x_l, x_u, H_type, H_ne, H_row, H_col, H_ptr, options=options)
    x0 = np.array([0., 0., 0.])
    x, g = bgo.solve(n, H_ne, x0, eval_f, eval_g, eval_h)
    inform = bgo.information()
    bgo.terminate()
    return x, g, inform


def test_bgo_status(solution):
    x, g, inform = solution
    assert inform['status'] == 0


def test_bgo_solution(solution):
    x, g, inform = solution
    assert np.all(x_l <= x) and np.all(x <= x_u)
    # returned values are consistent with the objective at x
    assert g == pytest.approx(eval_g(x), abs=1e-8)
    assert inform['obj'] == pytest.approx(eval_f(x))
    # first-order optimality: x_2 is on its upper bound with g_2 < 0
    # (a positive multiplier), and the free components are stationary
    assert x[2] == pytest.approx(x_u[2], abs=1e-8)
    assert g[2] < 0.0
    assert g[:2] == pytest.approx(np.zeros(2), abs=1e-5)
    # second-order: the Hessian on the free variables x_0, x_1 is
    # positive definite
    H = np.zeros((n, n))
    H[H_row, H_col] = eval_h(x)
    H = H + np.tril(H, -1).T
    assert np.all(np.linalg.eigvalsh(H[:2, :2]) > 0.0)
    assert x == pytest.approx(x_star, abs=1e-5)
    assert inform['obj'] == pytest.approx(f_star, abs=1e-8)
