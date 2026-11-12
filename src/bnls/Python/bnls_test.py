from galahad import bnls
import numpy as np
import pytest


# test problem:
# minimize 1/2 sum_i w_i r_i(x)^2 over [x_l,x_u], where
# r(x) = (x_0 x_1 - p, x_1 x_2 - 1, x_2 x_3 - 1, x_3 x_4 - 1)
p = 4.0
n = 5
m_r = 4
x_l = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
x_u = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
w = np.array([1.0, 1.0, 1.0, 1.0])

# Jacobian sparsity (coordinate format)
Jr_row = np.array([0, 0, 1, 1, 2, 2, 3, 3])
Jr_col = np.array([0, 1, 1, 2, 2, 3, 3, 4])

def eval_r(x):
    return np.array([x[0] * x[1] - p, x[1] * x[2] - 1.0,
                     x[2] * x[3] - 1.0, x[3] * x[4] - 1.0])

def eval_jr(x):
    return np.array([x[1], x[0], x[2], x[1], x[3], x[2], x[4], x[3]])

def jacobian(x):
    J = np.zeros((m_r, n))
    J[Jr_row, Jr_col] = eval_jr(x)
    return J

def eval_jr_dense(x):
    return jacobian(x).ravel()

def hessian(x):
    # Hessian J^T W J + sum_i w_i r_i H_i of the objective, where the only
    # nonzeros in each H_i are (H_i)_{i,i+1} = (H_i)_{i+1,i} = 1
    J = jacobian(x)
    H = J.T @ (w[:, None] * J)
    wr = w * eval_r(x)
    for i in range(m_r):
        H[i, i + 1] += wr[i]
        H[i + 1, i] += wr[i]
    return H

# bnls is a local method, so it is only required to find a local minimizer.
# Here the minimizer is unique: in the box, r_0 = x_0 x_1 - p <= 1 - p, so
# the objective is at least (p - 1)^2 / 2, with equality only when
# x_0 = x_1 = 1, and then r_1 = r_2 = r_3 = 0 give x_2 = x_3 = x_4 = 1.
# The bounds on x_0 and x_1 are strongly active (multipliers 1 - p), while
# those on x_2, x_3 and x_4 are weakly active (zero multipliers).
x_star = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
r_star = np.array([1.0 - p, 0.0, 0.0, 0.0])
z_star = np.array([1.0 - p, 1.0 - p, 0.0, 0.0, 0.0])
f_star = 0.5 * (p - 1.0)**2

@pytest.fixture(params=['coordinate', 'dense'])
def solution(request):
    Jr_type = request.param
    if Jr_type == 'coordinate':
        Jr_ne, Jr_r, Jr_c, eval_j = len(Jr_row), Jr_row, Jr_col, eval_jr
    else:
        Jr_ne, Jr_r, Jr_c, eval_j = m_r * n, None, None, eval_jr_dense
    options = bnls.initialize()
    options['jacobian_available'] = 2
    options['stop_pg_absolute'] = 0.00001
    options['blls_options']['sbls_options']['symmetric_linear_solver'] \
        = 'sytr '
    options['blls_options']['sbls_options']['definite_linear_solver'] \
        = 'potr '
    bnls.load(n, m_r, Jr_type, Jr_ne, Jr_r, Jr_c, 0, None, options)
    x0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    x, z, r, g, x_stat = bnls.solve(n, m_r, x_l, x_u, x0,
                                    eval_r, Jr_ne, eval_j, w)
    inform = bnls.information()
    bnls.terminate()
    return x, z, r, g, x_stat, inform


def test_bnls_status(solution):
    x, z, r, g, x_stat, inform = solution
    assert inform['status'] == 0


def test_bnls_solution(solution):
    x, z, r, g, x_stat, inform = solution
    assert np.all(x_l <= x) and np.all(x <= x_u)
    # returned values are consistent with the residuals at x
    assert r == pytest.approx(eval_r(x), abs=1e-8)
    assert g == pytest.approx(jacobian(x).T @ (w * r), abs=1e-8)
    assert inform['obj'] == pytest.approx(0.5 * np.sum(w * eval_r(x)**2))
    # first-order optimality: g = z, with z <= 0 for the variables on
    # their upper bounds; x_0 and x_1 are strongly active (z < 0, and
    # x_stat > 0 flags the upper bound as active)
    assert z == pytest.approx(g, abs=1e-5)
    assert x[:2] == pytest.approx(x_u[:2], abs=1e-8)
    assert np.all(z[:2] < 0.0)
    assert np.all(x_stat[:2] > 0)
    # second-order: the Hessian on the variables x_2, x_3, x_4 that are
    # not strongly active is positive definite
    assert np.all(np.linalg.eigvalsh(hessian(x)[2:, 2:]) > 0.0)
    assert x == pytest.approx(x_star, abs=1e-5)
    assert r == pytest.approx(r_star, abs=1e-5)
    assert z == pytest.approx(z_star, abs=1e-5)
    assert inform['obj'] == pytest.approx(f_star, abs=1e-8)
