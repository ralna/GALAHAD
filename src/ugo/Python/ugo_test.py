from galahad import ugo
import numpy as np
import pytest


# test problem: minimize x^2 cos(a x) over [x_l, x_u]
x_l = -1
x_u = 2
def eval_fgh(x):
    a = 10
    f = x * x * np.cos(a * x)
    g = - a * x * x * np.sin(a * x) + 2.0 * x * np.cos(a * x)
    h = - a * a * x * x * np.cos(a * x) - 4.0 * a * x * np.sin(a * x) \
        + 2.0 * np.cos(a * x)
    return f, g, h

# global minimizer and minimum, computed by Newton's method on g(x) = 0
x_star = 1.583361141494772
f_star = -2.487268692891855

@pytest.fixture
def solution():
    options = ugo.initialize()
    ugo.load(x_l, x_u, options=options)
    x, f, g, h = ugo.solve(eval_fgh)
    inform = ugo.information()
    ugo.terminate()
    return x, f, g, h, inform


def test_ugo_status(solution):
    x, f, g, h, inform = solution
    assert inform['status'] == 0


def test_ugo_solution(solution):
    x, f, g, h, inform = solution
    assert x_l <= x <= x_u
    # returned values are consistent with the objective at x
    assert (f, g, h) == pytest.approx(eval_fgh(x))
    # the global minimizer is an interior stationary point with h > 0
    assert g == pytest.approx(0.0, abs=1e-5)
    assert h > 0.0
    assert x == pytest.approx(x_star, abs=1e-5)
    assert f == pytest.approx(f_star, abs=1e-8)
