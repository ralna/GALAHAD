from galahad import arc
import numpy as np
import pytest


# test problem:
# minimize (x_0 + x_2 + p)^2 + (x_1 + x_2)^2 + cos(x_0)
p = 4
n = 3

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

# arc is a local method, so it is only required to find a local minimizer.
# Stationary points satisfy x_1 = -x_2, x_0 + x_2 + p = 0 and sin(x_0) = 0,
# and they are local minimizers when x_0 = (2k+1) pi, x_2 = -p - x_0 and
# x_1 = -x_2, all with f = -1 (these are in fact global minimizers, as
# f >= -1). From the starting point (1, 1, 1), arc converges to the one
# with k = -1.
x_star = np.array([-np.pi, p - np.pi, np.pi - p])
f_star = -1.0

@pytest.fixture
def solution():
    options = arc.initialize()
    arc.load(n, H_type, H_ne, H_row, H_col, H_ptr, options=options)
    x0 = np.array([1., 1., 1.])
    x, g = arc.solve(n, H_ne, x0, eval_f, eval_g, eval_h)
    inform = arc.information()
    arc.terminate()
    return x, g, inform


def test_arc_status(solution):
    x, g, inform = solution
    assert inform['status'] == 0


def test_arc_solution(solution):
    x, g, inform = solution
    # returned values are consistent with the objective at x
    assert g == pytest.approx(eval_g(x), abs=1e-8)
    assert inform['obj'] == pytest.approx(eval_f(x))
    # first-order optimality
    assert g == pytest.approx(np.zeros(n), abs=1e-5)
    # second-order: the Hessian is positive definite
    H = np.zeros((n, n))
    H[H_row, H_col] = eval_h(x)
    H = H + np.tril(H, -1).T
    assert np.all(np.linalg.eigvalsh(H) > 0.0)
    assert x == pytest.approx(x_star, abs=1e-5)
    assert inform['obj'] == pytest.approx(f_star, abs=1e-8)
