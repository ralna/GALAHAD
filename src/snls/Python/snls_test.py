from galahad import snls
import numpy as np
import pytest


# test problem:
# minimize 1/2 sum_i w_i r_i(x)^2 over the intersection of the simplices
# x_0 + x_3 = 1, x_1 + x_4 = 1, x_0, x_1, x_3, x_4 >= 0
# (x_2 is unconstrained), where
# r(x) = (x_0 x_1 - p, x_1 x_2 - 1, x_2 x_3 - 1, x_3 x_4 - 1)
p = 4.0
n = 5
m_r = 4
w = np.array([1.0, 1.0, 1.0, 1.0])

# cohorts: x_0, x_3 in cohort 0, x_1, x_4 in cohort 1, x_2 in none
m_c = 2
cohort = np.array([0, 1, -1, 0, 1])
E = np.zeros((n, m_c))
for i, c in enumerate(cohort):
    if c >= 0:
        E[i, c] = 1.0

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

# snls is a local method, so it is only required to find a local minimizer.
# From the starting point (0.5, ..., 0.5), it converges to x = (1, 1, 1, 0, 0),
# where x_2 = 1 makes r_1 = 0. The dual optimality conditions
# J^T W r = E y + z, with z >= 0 and x^T z = 0, then hold with
# y = (1 - p, 1 - p) and z = (0, 0, 0, p - 2, p - 1), so the bounds
# x_3, x_4 >= 0 are strongly active. Every feasible direction d that keeps
# them active has d_0 = d_1 = d_3 = d_4 = 0, along which the objective is
# strictly convex in x_2, so x is a strict local minimizer.
x_star = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
r_star = np.array([1.0 - p, 0.0, -1.0, -1.0])
y_star = np.array([1.0 - p, 1.0 - p])
z_star = np.array([0.0, 0.0, 0.0, p - 2.0, p - 1.0])
f_star = 0.5 * ((p - 1.0)**2 + 2.0)

# with stop_pg_absolute = 1e-5 as in test_snls.py, x_2 is only accurate
# to about 2e-5
tol = 1e-4

@pytest.fixture(params=['coordinate', 'dense'])
def solution(request):
    Jr_type = request.param
    if Jr_type == 'coordinate':
        Jr_ne, Jr_r, Jr_c, eval_j = len(Jr_row), Jr_row, Jr_col, eval_jr
    else:
        Jr_ne, Jr_r, Jr_c, eval_j = m_r * n, None, None, eval_jr_dense
    options = snls.initialize()
    options['jacobian_available'] = 2
    options['stop_pg_absolute'] = 0.00001
    options['slls_options']['sbls_options']['symmetric_linear_solver'] \
        = 'sytr '
    options['slls_options']['sbls_options']['definite_linear_solver'] \
        = 'potr '
    snls.load(n, m_r, m_c, Jr_type, Jr_ne, Jr_r, Jr_c, 0, None,
              cohort, options)
    x0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    x, y, z, r, g, x_stat = snls.solve(n, m_r, m_c, x0, eval_r, Jr_ne,
                                       eval_j, w)
    inform = snls.information()
    snls.terminate()
    return x, y, z, r, g, x_stat, inform


def test_snls_status(solution):
    x, y, z, r, g, x_stat, inform = solution
    assert inform['status'] == 0


def test_snls_solution(solution):
    x, y, z, r, g, x_stat, inform = solution
    # x is feasible
    assert E.T @ x == pytest.approx(np.ones(m_c), abs=1e-8)
    assert np.all(x[cohort >= 0] >= 0.0)
    # returned values are consistent with the residuals at x
    assert r == pytest.approx(eval_r(x), abs=1e-8)
    assert g == pytest.approx(jacobian(x).T @ (w * r), abs=1e-8)
    assert inform['obj'] == pytest.approx(0.5 * np.sum(w * eval_r(x)**2))
    # first-order optimality, with multipliers computed from g (see
    # test_snls_multipliers): x_0, x_1 > 0 give z_0 = z_1 = 0, so y = g_0:1,
    # x_2 is unconstrained so g_2 = 0, and z_3, z_4 > 0 on the active
    # bounds x_3 = x_4 = 0
    assert x[3:] == pytest.approx(np.zeros(2), abs=1e-8)
    assert np.all(x_stat[3:] < 0)
    y_g = g[:2]
    z_g = g - E @ y_g
    assert z_g[2] == pytest.approx(0.0, abs=tol)
    assert np.all(z_g[3:] > 0.0)
    # second-order: the only direction keeping the strongly active bounds
    # active is along x_2, on which the Hessian is positive
    assert hessian(x)[2, 2] > 0.0
    assert x == pytest.approx(x_star, abs=tol)
    assert r == pytest.approx(r_star, abs=tol)
    assert y_g == pytest.approx(y_star, abs=tol)
    assert z_g == pytest.approx(z_star, abs=tol)
    assert inform['obj'] == pytest.approx(f_star, abs=1e-8)


@pytest.mark.xfail(strict=True, reason='snls returns the y and z of its '
                   'last slls subproblem, for which slls returns y = 0 and '
                   'z = g, violating z >= 0 and x^T z = 0 (see '
                   'https://github.com/ralna/GALAHAD/issues/583)')
def test_snls_multipliers(solution):
    x, y, z, r, g, x_stat, inform = solution
    assert g == pytest.approx(E @ y + z, abs=tol)
    assert y == pytest.approx(y_star, abs=tol)
    assert z == pytest.approx(z_star, abs=tol)
