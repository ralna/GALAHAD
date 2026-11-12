from galahad import trb
import numpy as np
import pytest


# test problem:
# minimize (x_0 + x_2 + p)^2 + (x_1 + x_2)^2 + cos(x_0)
# over [x_l,x_u]
p = 4
n = 3
x_l = np.array([-np.inf, -np.inf, 0.0])
x_u = np.array([1.1, 1.1, 1.1])

# Hessian sparsity (lower triangle, coordinate format)
H_type = 'coordinate'
H_ne = 5
H_row = np.array([0, 2, 1, 2, 2])
H_col = np.array([0, 0, 1, 1, 2])
H_ptr = None

def eval_f(x):
    return (x[0] + x[2] + p)**2 + (x[1] + x[2])**2 + np.cos(x[0])

def eval_g(x):
    return np.array([2. * (x[0] + x[2] + p) - np.sin(x[0]),
                     2. * (x[1] + x[2]),
                     2. * (x[0] + x[2] + p) + 2. * (x[1] + x[2])])

def eval_h(x):
    return np.array([2. - np.cos(x[0]), 2., 2., 2., 4.])

# trb is a local method, so it is only required to find a local minimizer.
# From the starting point (1, 1, 1), it converges to the one with x_2 on
# its lower bound 0. Then g_1 = 0 gives x_1 = 0, and x_0 solves
# 2 (x_0 + p) - sin(x_0) = 0, computed by Newton's method (the root is
# unique, as the left-hand side is strictly increasing).
x_star = np.array([-3.7246927803094874, 0.0, 0.0])
f_star = -0.75896562424494

@pytest.fixture
def solution():
    options = trb.initialize()
    trb.load(n, H_type, H_ne, H_row, H_col, H_ptr, options=options)
    x0 = np.array([1., 1., 1.])
    x, g = trb.solve(n, H_ne, x_l, x_u, x0, eval_f, eval_g, eval_h)
    inform = trb.information()
    trb.terminate()
    return x, g, inform


def test_trb_status(solution):
    x, g, inform = solution
    assert inform['status'] == 0


def test_trb_solution(solution):
    x, g, inform = solution
    assert np.all(x_l <= x) and np.all(x <= x_u)
    # returned values are consistent with the objective at x
    assert g == pytest.approx(eval_g(x), abs=1e-8)
    assert inform['obj'] == pytest.approx(eval_f(x))
    # first-order optimality: x_2 is on its lower bound with g_2 > 0
    # (a positive multiplier), and the free components are stationary
    assert x[2] == pytest.approx(x_l[2], abs=1e-8)
    assert g[2] > 0.0
    assert g[:2] == pytest.approx(np.zeros(2), abs=1e-5)
    # second-order: the Hessian on the free variables x_0, x_1 is
    # positive definite
    H = np.zeros((n, n))
    H[H_row, H_col] = eval_h(x)
    H = H + np.tril(H, -1).T
    assert np.all(np.linalg.eigvalsh(H[:2, :2]) > 0.0)
    assert x == pytest.approx(x_star, abs=1e-5)
    assert inform['obj'] == pytest.approx(f_star, abs=1e-8)
