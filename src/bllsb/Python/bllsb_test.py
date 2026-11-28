from galahad import bllsb
import numpy as np
import pytest


# test problem:
# minimize 1/2 || A_o x - b ||_W^2 + sigma/2 || x - x_s ||_2^2
# subject to x_l <= x <= x_u, where
#       ( 1 1 0 )
# A_o = ( 0 1 1 ), b = ( 2, 2, 3, 1 ), W = diag( 1, 1, 1, 2 ), sigma = 1,
#       ( 1 0 1 )
#       ( 0 1 0 )
# -1 <= x_0 <= 1, x_1 is free and x_2 <= 2
n = 3
o = 4
sigma = 1.0
infinity = float("inf")
A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0],
              [0.0, 1.0, 0.0]])
b = np.array([2.0, 2.0, 3.0, 1.0])
w = np.array([1.0, 1.0, 1.0, 2.0])
x_l = np.array([-1.0, -infinity, -infinity])
x_u = np.array([1.0, infinity, 2.0])

def storage(Ao_type):
    # return Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr, Ao_val for A
    # in the given storage scheme
    if Ao_type in ('dense', 'dense_by_rows'):
        return o * n, None, None, 0, None, A.ravel()
    if Ao_type == 'dense_by_columns':
        return o * n, None, None, 0, None, A.ravel(order='F')
    if Ao_type == 'sparse_by_columns':
        cols, rows = np.nonzero(A.T)
    else:
        rows, cols = np.nonzero(A)
    vals = A[rows, cols]
    if Ao_type == 'coordinate':
        return len(vals), rows, cols, 0, None, vals
    if Ao_type == 'sparse_by_rows':
        ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=o))]
        return len(vals), None, cols, o + 1, ptr, vals
    ptr = np.r_[0, np.cumsum(np.bincount(cols, minlength=n))]
    return len(vals), rows, None, n + 1, ptr, vals

def gradient(x, x_s):
    x_s = np.zeros(n) if x_s is None else x_s
    return A.T @ (w * (A @ x - b)) + sigma * (x - x_s)

def objective(x, x_s):
    x_s = np.zeros(n) if x_s is None else x_s
    r = A @ x - b
    return 0.5 * np.sum(w * r**2) + 0.5 * sigma * np.sum((x - x_s)**2)

def solve(Ao_type, x_s):
    Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr, Ao_val = storage(Ao_type)
    options = bllsb.initialize()
    # use the LAPACK solver sytr rather than the default slblt, which needs
    # the environment variables OMP_CANCELLATION and OMP_PROC_BIND set.
    options['symmetric_linear_solver'] = 'sytr'
    options['fdc_options']['symmetric_linear_solver'] = 'sytr'
    bllsb.load(n, o, Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr,
               options)
    x0 = np.zeros(n)
    z0 = np.zeros(n)
    x, z, r, x_stat = bllsb.solve(n, o, Ao_ne, Ao_val, b, sigma, x_l, x_u,
                                  x0, z0, w, x_s)
    inform = bllsb.information()
    bllsb.terminate()
    return x, z, r, x_stat, inform

# bllsb is a local method, so it is only required to find a local minimizer,
# but here the objective is strictly convex (its Hessian
# H = A_o^T W A_o + sigma I = [[3, 1, 1], [1, 5, 1], [1, 1, 3]] is positive
# definite), so the minimizer is unique. With A_o^T W b = ( 5, 6, 5 ), H x =
# ( 5, 6, 5 ) + sigma x_s + z, where z_0 <= 0 at the upper bound x_0 = 1, and
# x_1, x_2 solve [[5, 1], [1, 3]] ( x_1, x_2 ) = ( 5, 4 ) + sigma x_s[1:].
# The run in test_bllsb.py uses x_s = 0.5 e, and we also run without shifts
RUNS = {
    'shift': dict(x_s=np.full(n, 0.5),
                  x=np.array([1.0, 6.0 / 7.0, 17.0 / 14.0]),
                  z=np.array([-3.0 / 7.0, 0.0, 0.0])),
    'no_shift': dict(x_s=None,
                     x=np.array([1.0, 11.0 / 14.0, 15.0 / 14.0]),
                     z=np.array([-1.0 / 7.0, 0.0, 0.0])),
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


def test_bllsb_status(solution):
    x, z, r, x_stat, inform = solution
    assert inform['status'] == 0


def test_bllsb_solution(run, solution):
    x, z, r, x_stat, inform = solution
    x_s = RUNS[run]['x_s']
    # returned values are consistent with the objective at x
    assert r == pytest.approx(A @ x - b, abs=1e-8)
    assert inform['obj'] == pytest.approx(objective(x, x_s))
    # first-order optimality: x is feasible, g = z, and z_j is zero, at
    # least zero or at most zero when x_j is free, on its lower bound or on
    # its upper bound respectively; x_stat > 0 flags the active upper bound
    g = gradient(x, x_s)
    assert np.all(x >= x_l) and np.all(x <= x_u)
    assert g == pytest.approx(z, abs=1e-6)
    assert z[1:] == pytest.approx(np.zeros(2), abs=1e-6)
    assert z[0] < 0.0
    assert x_stat[0] > 0 and np.all(x_stat[1:] == 0)
    assert x == pytest.approx(RUNS[run]['x'], abs=1e-6)
    assert z == pytest.approx(RUNS[run]['z'], abs=1e-6)
    assert inform['obj'] == pytest.approx(RUNS[run]['f'], abs=1e-8)
