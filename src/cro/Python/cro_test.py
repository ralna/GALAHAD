from galahad import cro
import numpy as np
import pytest


# cro takes a primal-dual solution (x, y, z) of the convex QP
# minimize 1/2 x^T H x + g^T x subject to c_l <= A x <= c_u, x_l <= x <= x_u,
# together with the active sets x_stat and c_stat, and moves the multipliers
# to a basic solution: one whose basic active constraints and bounds (status
# -1 or 1) are linearly independent, with zero multipliers for the remaining
# non-basic (dependent) active ones (status -2 or 2). x is unchanged.
#
# test problems:
# 1) as in test_cro.py, n = 11, H tridiagonal with 1 on the diagonal and
# 1/2 off it, and x = ( 0, 1, ..., 1 ) on all its lower bounds, so that
# H x + g = ( 1, ..., 1 ), and all three constraints active. The active
# bounds alone are independent, and cro makes them basic and the
# constraints non-basic, so y = 0 and z = H x + g = 1
# 2) minimize 1/2 ||x||^2 - 2 x_0 - 2 x_1 subject to x_0 + x_1 <= 2 and
# x <= 1, at x = ( 1, 1 ), where x + g = ( -1, -1 ) = y_0 ( 1, 1 ) + z with
# y_0 = t, z = ( - 1 - t, - 1 - t ), t in [ -1, 0 ]; starting from t = -1/2
# cro makes the bounds basic, so y = 0 and z = ( -1, -1 )
# 3) minimize 1/2 ||x||^2 - 3 x_0 - 2 x_1 subject to x_0 + x_1 <= 2,
# x_0 - x_1 >= 0 and x_0 <= 1, at x = ( 1, 1 ), where all three are active
# and x + g = ( -2, -1 ) = A^T y + z with y = ( t - 1, t ), z = ( - 1 - 2 t,
# 0 ), t in [ 0, 1 ]. The bound can't be non-basic (z_0 = 0 needs t = -1/2),
# so a constraint must stay basic; whatever t it starts from, cro keeps
# x_0 + x_1 <= 2, giving y = ( -1, 0 ) and z = ( -1, 0 )
# 4) as 3) with x_0 + x_1 = 2 an equality constraint (m_equal = 1)
# 5) minimize 1/2 ||x||^2 - 3 x_0 - x_1 subject to x_0 + x_1 <= 2 and
# x <= 5, whose solution x = ( 2, 0 ), y = -1, z = 0 is already basic
# 6) the linear program minimize - x_0 - x_1 subject to x_0 + x_1 <= 2 and
# 0 <= x <= 1, at x = ( 1, 1 ) with y = -1/2 and z = ( -1/2, -1/2 ); as for
# 2), cro makes the bounds basic, so y = 0 and z = ( -1, -1 )
infinity = float("inf")
H_example = (np.eye(11) + 0.5 * np.eye(11, k=1) + 0.5 * np.eye(11, k=-1))
A_example = np.vstack([np.ones(11), np.r_[0.0, 0.0, np.ones(9)],
                       np.r_[0.0, np.ones(10)]])
A_2 = np.array([[1.0, 1.0]])
A_3 = np.array([[1.0, 1.0], [1.0, -1.0]])

def forced(t, m_equal=0):
    c_l = np.array([2.0 if m_equal else -infinity, 0.0])
    return dict(H=np.eye(2), g=np.array([-3.0, -2.0]), A=A_3,
                m_equal=m_equal, c_l=c_l, c_u=np.array([2.0, infinity]),
                x_l=np.array([-infinity, -infinity]),
                x_u=np.array([1.0, infinity]), x=np.array([1.0, 1.0]),
                y=np.array([t - 1.0, t]), z=np.array([-1.0 - 2.0 * t, 0.0]),
                x_stat=[1, 0], c_stat=[1, -1],
                y_star=np.array([-1.0, 0.0]), z_star=np.array([-1.0, 0.0]),
                x_stat_star=[1, 0], c_stat_star=[1, -2])

PROBLEMS = {
    'example': dict(H=H_example,
                    g=np.r_[0.5, -0.5, -np.ones(8), -0.5], A=A_example,
                    m_equal=1, c_l=np.array([10.0, 9.0, -infinity]),
                    c_u=np.array([10.0, infinity, 10.0]),
                    x_l=np.r_[0.0, np.ones(10)], x_u=np.full(11, infinity),
                    x=np.r_[0.0, np.ones(10)], y=np.array([-1.0, 1.5, -2.0]),
                    z=np.r_[2.0, 4.0, np.full(9, 2.5)],
                    x_stat=[-1] * 11, c_stat=[-1, -1, 1],
                    y_star=np.zeros(3), z_star=np.ones(11),
                    x_stat_star=[-1] * 11, c_stat_star=[-2, -2, 2]),
    'bounds': dict(H=np.eye(2), g=np.array([-2.0, -2.0]), A=A_2, m_equal=0,
                   c_l=np.array([-infinity]), c_u=np.array([2.0]),
                   x_l=np.array([-infinity, -infinity]),
                   x_u=np.array([1.0, 1.0]), x=np.array([1.0, 1.0]),
                   y=np.array([-0.5]), z=np.array([-0.5, -0.5]),
                   x_stat=[1, 1], c_stat=[1],
                   y_star=np.array([0.0]), z_star=np.array([-1.0, -1.0]),
                   x_stat_star=[1, 1], c_stat_star=[2]),
    'forced_0': forced(0.0),
    'forced_1/2': forced(0.5),
    'forced_1': forced(1.0),
    'equality': forced(0.5, m_equal=1),
    'basic': dict(H=np.eye(2), g=np.array([-3.0, -1.0]), A=A_2, m_equal=0,
                  c_l=np.array([-infinity]), c_u=np.array([2.0]),
                  x_l=np.array([-infinity, -infinity]),
                  x_u=np.array([5.0, 5.0]), x=np.array([2.0, 0.0]),
                  y=np.array([-1.0]), z=np.zeros(2),
                  x_stat=[0, 0], c_stat=[1],
                  y_star=np.array([-1.0]), z_star=np.zeros(2),
                  x_stat_star=[0, 0], c_stat_star=[1]),
    'linear': dict(H=np.zeros((2, 2)), g=np.array([-1.0, -1.0]), A=A_2,
                   m_equal=0, c_l=np.array([-infinity]), c_u=np.array([2.0]),
                   x_l=np.zeros(2), x_u=np.ones(2), x=np.array([1.0, 1.0]),
                   y=np.array([-0.5]), z=np.array([-0.5, -0.5]),
                   x_stat=[1, 1], c_stat=[1],
                   y_star=np.array([0.0]), z_star=np.array([-1.0, -1.0]),
                   x_stat_star=[1, 1], c_stat_star=[2]),
}

def sparse_by_rows(M):
    # return M_ne, M_val, M_col, M_ptr for M stored by rows
    rows, cols = np.nonzero(M)
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=M.shape[0]))]
    return len(rows), M[rows, cols], cols, ptr

def cro_options(**extra_options):
    options = cro.initialize()
    options['print_level'] = 0
    # use the LAPACK solvers rather than the defaults, which (for slblt)
    # need the environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    options['symmetric_linear_solver'] = 'sytr'
    options['unsymmetric_linear_solver'] = 'getr'
    options['sbls_options']['symmetric_linear_solver'] = 'sytr'
    options['sbls_options']['definite_linear_solver'] = 'sytr'
    options['sbls_options']['unsymmetric_linear_solver'] = 'getr'
    options.update(extra_options)
    return options

def crossover(p, **extra_options):
    n, m = len(p['g']), p['A'].shape[0]
    # H is given by its lower triangle
    H_ne, H_val, H_col, H_ptr = sparse_by_rows(np.tril(p['H']))
    A_ne, A_val, A_col, A_ptr = sparse_by_rows(p['A'])
    options = cro_options(**extra_options)
    # x, c, y, z, x_stat and c_stat are updated in place, so pass copies
    x, c, y, z, x_stat, c_stat, inform \
        = cro.crossover_solution(n, m, p['m_equal'], p['g'], H_ne, H_val,
                                 H_col, H_ptr, A_ne, A_val, A_col, A_ptr,
                                 p['c_l'], p['c_u'], p['x_l'], p['x_u'],
                                 p['x'].copy(), p['A'] @ p['x'],
                                 p['y'].copy(), p['z'].copy(),
                                 np.array(p['x_stat']),
                                 np.array(p['c_stat']),
                                 options)
    cro.terminate()
    return x, c, y, z, x_stat, c_stat, inform

@pytest.fixture(params=list(PROBLEMS))
def solution(request):
    problem = request.param
    return problem, crossover(PROBLEMS[problem])


def test_cro_status(solution):
    problem, (x, c, y, z, x_stat, c_stat, inform) = solution
    assert inform['status'] == 0


def test_cro_solution(solution):
    # x is unchanged, c = A x, the first-order conditions H x + g = A^T y + z
    # hold, and y and z are the expected basic multipliers
    problem, (x, c, y, z, x_stat, c_stat, inform) = solution
    p = PROBLEMS[problem]
    assert x == pytest.approx(p['x'], abs=1e-8)
    assert c == pytest.approx(p['A'] @ x, abs=1e-8)
    assert p['H'] @ x + p['g'] == pytest.approx(p['A'].T @ y + z, abs=1e-8)
    assert y == pytest.approx(p['y_star'], abs=1e-8)
    assert z == pytest.approx(p['z_star'], abs=1e-8)


def test_cro_basis(solution):
    # the status arrays are as expected; inactive and non-basic constraints
    # and bounds have zero multipliers, the basic ones are linearly
    # independent and the number of non-basic ones is inform['dependent']
    problem, (x, c, y, z, x_stat, c_stat, inform) = solution
    p = PROBLEMS[problem]
    assert list(x_stat) == p['x_stat_star']
    assert list(c_stat) == p['c_stat_star']
    n = len(x)
    for stat, mult in [(x_stat, z), (c_stat, y)]:
        assert np.all(mult[np.abs(stat) != 1] == 0.0)
    basic = np.vstack([p['A'][np.abs(c_stat) == 1],
                       np.eye(n)[np.abs(x_stat) == 1]])
    assert np.linalg.matrix_rank(basic) == basic.shape[0]
    assert inform['dependent'] == (np.sum(np.abs(x_stat) == 2)
                                   + np.sum(np.abs(c_stat) == 2))


def test_cro_in_place():
    # x, c, y, z, x_stat and c_stat are updated in place and returned
    p = PROBLEMS['example']
    n, m = len(p['g']), p['A'].shape[0]
    H_ne, H_val, H_col, H_ptr = sparse_by_rows(np.tril(p['H']))
    A_ne, A_val, A_col, A_ptr = sparse_by_rows(p['A'])
    options = cro_options()
    arrays = [p['x'].copy(), p['A'] @ p['x'], p['y'].copy(), p['z'].copy(),
              np.array(p['x_stat']), np.array(p['c_stat'])]
    output = cro.crossover_solution(n, m, p['m_equal'], p['g'], H_ne, H_val,
                                    H_col, H_ptr, A_ne, A_val, A_col, A_ptr,
                                    p['c_l'], p['c_u'], p['x_l'], p['x_u'],
                                    *arrays, options)
    cro.terminate()
    for array, returned in zip(arrays, output[:6]):
        assert returned is array
    assert arrays[2] == pytest.approx(p['y_star'], abs=1e-8)
    assert arrays[3] == pytest.approx(p['z_star'], abs=1e-8)
    assert list(arrays[4]) == p['x_stat_star']
    assert list(arrays[5]) == p['c_stat_star']


def test_cro_check_io():
    # checking the input and output gives the same result
    p = PROBLEMS['example']
    x, c, y, z, x_stat, c_stat, inform = crossover(p, check_io=True)
    assert inform['status'] == 0
    assert y == pytest.approx(p['y_star'], abs=1e-8)
    assert z == pytest.approx(p['z_star'], abs=1e-8)


def test_cro_bad_bounds():
    # when the input is checked, x_l > x_u is reported as status -4
    p = dict(PROBLEMS['example'], x_u=np.full(11, -1.0))
    with pytest.warns(RuntimeWarning):
        x, c, y, z, x_stat, c_stat, inform = crossover(p, check_io=True)
    assert inform['status'] == -4

