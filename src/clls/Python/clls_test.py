from galahad import clls
import numpy as np
import pytest


# test problem:
# minimize 1/2 || A_o x - b ||_W^2 + sigma/2 || x - x_s ||_2^2
# subject to c_l <= A x <= c_u and x_l <= x <= x_u, where
#       ( 1 1 0 )
# A_o = ( 0 1 1 ), b = ( 2, 2, 3, 1 ), W = diag( 1, 1, 1, 2 ), sigma = 1,
#       ( 1 0 1 )
#       ( 0 1 0 )
# A = ( 2 1 0 ), c_l = ( 1, 2 ), c_u = ( 2, 2 ),
#     ( 0 1 1 )
# -1 <= x_0 <= 1, x_1 is free and x_2 <= 2
n = 3
o = 4
m = 2
sigma = 1.0
infinity = float("inf")
Ao = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0],
               [0.0, 1.0, 0.0]])
A = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
b = np.array([2.0, 2.0, 3.0, 1.0])
w = np.array([1.0, 1.0, 1.0, 2.0])
c_l = np.array([1.0, 2.0])
c_u = np.array([2.0, 2.0])
x_l = np.array([-1.0, -infinity, -infinity])
x_u = np.array([1.0, infinity, 2.0])

def storage(M, M_type):
    # return M_ne, M_row, M_col, M_ptr_ne, M_ptr, M_val for the matrix M
    # in the given storage scheme
    rows_M, cols_M = M.shape
    if M_type in ('dense', 'dense_by_rows'):
        return M.size, None, None, 0, None, M.ravel()
    if M_type == 'dense_by_columns':
        return M.size, None, None, 0, None, M.ravel(order='F')
    if M_type == 'sparse_by_columns':
        cols, rows = np.nonzero(M.T)
    else:
        rows, cols = np.nonzero(M)
    vals = M[rows, cols]
    if M_type == 'coordinate':
        return len(vals), rows, cols, 0, None, vals
    if M_type == 'sparse_by_rows':
        ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=rows_M))]
        return len(vals), None, cols, rows_M + 1, ptr, vals
    ptr = np.r_[0, np.cumsum(np.bincount(cols, minlength=cols_M))]
    return len(vals), rows, None, cols_M + 1, ptr, vals

def gradient(x, x_s):
    x_s = np.zeros(n) if x_s is None else x_s
    return Ao.T @ (w * (Ao @ x - b)) + sigma * (x - x_s)

def objective(x, x_s):
    x_s = np.zeros(n) if x_s is None else x_s
    r = Ao @ x - b
    return 0.5 * np.sum(w * r**2) + 0.5 * sigma * np.sum((x - x_s)**2)

def solve(M_type, x_s):
    Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr, Ao_val = storage(Ao, M_type)
    A_ne, A_row, A_col, A_ptr_ne, A_ptr, A_val = storage(A, M_type)
    options = clls.initialize()
    # use the LAPACK solver sytr rather than the default slblt, which needs
    # the environment variables OMP_CANCELLATION and OMP_PROC_BIND set.
    options['symmetric_linear_solver'] = 'sytr'
    options['fdc_options']['symmetric_linear_solver'] = 'sytr'
    clls.load(n, o, m, M_type, Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr,
              M_type, A_ne, A_row, A_col, A_ptr_ne, A_ptr, options)
    x0 = np.zeros(n)
    y0 = np.zeros(m)
    z0 = np.zeros(n)
    x, y, z, r, c, x_stat, c_stat = clls.solve(n, o, m, Ao_ne, Ao_val, b,
                                               sigma, A_ne, A_val, c_l, c_u,
                                               x_l, x_u, x0, y0, z0, w, x_s)
    inform = clls.information()
    clls.terminate()
    return x, y, z, r, c, x_stat, c_stat, inform

# clls is a local method, so it is only required to find a local minimizer,
# but here the objective is strictly convex (its Hessian
# H = A_o^T W A_o + sigma I = [[3, 1, 1], [1, 5, 1], [1, 1, 3]] is positive
# definite), so the minimizer is unique. Enumerating the active sets, the
# only KKT point has the first constraint active at its upper bound
# 2 x_0 + x_1 = 2, the equality x_1 + x_2 = 2, and no active bounds, so
# z = 0 and H x - A_o^T W b - sigma x_s = A^T y with y_0 <= 0.
# The run in test_clls.py uses x_s = 0.5 e, and we also run without shifts
RUNS = {
    'shift': dict(x_s=np.full(n, 0.5),
                  x=np.array([35.0 / 54.0, 19.0 / 27.0, 35.0 / 27.0]),
                  y=np.array([-7.0 / 9.0, -7.0 / 27.0])),
    'no_shift': dict(x_s=None,
                     x=np.array([17.0 / 27.0, 20.0 / 27.0, 34.0 / 27.0]),
                     y=np.array([-5.0 / 9.0, 4.0 / 27.0])),
}
for run in RUNS.values():
    run['f'] = objective(run['x'], run['x_s'])

STORAGE = ['coordinate', 'sparse_by_rows', 'sparse_by_columns',
           'dense_by_rows', 'dense_by_columns', 'dense']

@pytest.fixture(params=RUNS.keys())
def run(request):
    return request.param

@pytest.fixture(params=STORAGE)
def solution(request, run):
    return solve(request.param, RUNS[run]['x_s'])


def test_clls_status(solution):
    x, y, z, r, c, x_stat, c_stat, inform = solution
    assert inform['status'] == 0


def test_clls_solution(run, solution):
    x, y, z, r, c, x_stat, c_stat, inform = solution
    x_s = RUNS[run]['x_s']
    # x is feasible, c = A x, and the first-order conditions g = A^T y + z
    # hold, with y_0 <= 0 for the constraint active at its upper bound, and
    # z = 0 as no bounds are active; c_stat > 0 flags the upper bound
    assert c == pytest.approx(A @ x, abs=1e-8)
    assert np.all(x >= x_l) and np.all(x <= x_u)
    assert np.all(c >= c_l - 1e-8) and np.all(c <= c_u + 1e-8)
    assert c == pytest.approx(c_u, abs=1e-8)
    assert gradient(x, x_s) == pytest.approx(A.T @ y + z, abs=1e-6)
    assert y[0] < 0.0
    assert z == pytest.approx(np.zeros(n), abs=1e-6)
    assert np.all(x_stat == 0) and c_stat[0] > 0 and c_stat[1] != 0
    assert x == pytest.approx(RUNS[run]['x'], abs=1e-6)
    assert y == pytest.approx(RUNS[run]['y'], abs=1e-6)


def test_clls_residuals(run, solution):
    x, y, z, r, c, x_stat, c_stat, inform = solution
    x_s = RUNS[run]['x_s']
    assert r == pytest.approx(Ao @ x - b, abs=1e-8)
    assert inform['obj'] == pytest.approx(objective(x, x_s), abs=1e-8)
