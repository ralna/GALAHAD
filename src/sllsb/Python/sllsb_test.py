from galahad import sllsb
import numpy as np
import pytest


# test problem:
# minimize 1/2 || A_o x - b ||_W^2 + sigma/2 || x - x_s ||_2^2
# over the simplex e^T x = 1, x >= 0, where
# A_o = (  I  ) and b = ( 1, 2, ..., n )
#       ( e^T )         (     n + 1    )
n = 10
o = n + 1
m = 1
sigma = 1.0
A = np.vstack([np.eye(n), np.ones((1, n))])
b = np.r_[np.arange(1, n + 1), n + 1].astype(float)

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

def gradient(x, w=None, x_s=None):
    w = np.ones(o) if w is None else w
    x_s = np.zeros(n) if x_s is None else x_s
    return A.T @ (w * (A @ x - b)) + sigma * (x - x_s)

def objective(x, w=None, x_s=None):
    w = np.ones(o) if w is None else w
    x_s = np.zeros(n) if x_s is None else x_s
    r = A @ x - b
    return 0.5 * np.sum(w * r**2) + 0.5 * sigma * np.sum((x - x_s)**2)

def solve(Ao_type, cohort, w, x_s):
    Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr, Ao_val = storage(Ao_type)
    options = sllsb.initialize()
    # use the LAPACK solver sytr rather than the default slblt, which needs
    # the environment variables OMP_CANCELLATION and OMP_PROC_BIND set.
    options['symmetric_linear_solver'] = 'sytr'
    options['fdc_options']['symmetric_linear_solver'] = 'sytr'
    sllsb.load(n, o, m, Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr,
               cohort, options)
    x0 = np.zeros(n)
    y0 = np.zeros(m)
    z0 = np.zeros(n)
    x, y, z, r, x_stat = sllsb.solve(n, o, m, Ao_ne, Ao_val, b, sigma,
                                     x0, y0, z0, w, x_s)
    inform = sllsb.information()
    sllsb.terminate()
    return x, y, z, r, x_stat, inform

def check_optimality(x, g, tol=1e-6, feas_tol=1e-8):
    # x is feasible, and the first-order conditions g = e y + z, z >= 0,
    # x^T z = 0 hold for y = min_j g_j. As the objective is strictly convex,
    # x is then the unique minimizer
    assert np.sum(x) == pytest.approx(1.0, abs=feas_tol)
    assert np.all(x >= 0.0)
    z = g - np.min(g)
    assert abs(x @ z) <= tol

# sllsb is a local method, so it is only required to find a local minimizer,
# but here the objective is strictly convex, so the minimizer is unique.
# On the simplex, the last residual e^T x - (n + 1) = -n is constant and
# g_j = (1 + sigma) x_j - (j + 1) - n, so the free components satisfy
# x_j = (y + j + 1 + n) / (1 + sigma). With sigma = 1, only the last two
# components are free, giving the values below, with y = -18.5 and
# z_j = 7.5 - j for j < 8.
x_star = np.r_[np.zeros(n - 2), 0.25, 0.75]
y_star = np.array([-18.5])
z_star = np.r_[7.5 - np.arange(n - 2), 0.0, 0.0]
r_star = A @ x_star - b
f_star = objective(x_star)

# the two runs in test_sllsb.py: an implicit single cohort with default
# weights and shifts, and an explicit cohort with unit weights and zero shifts
RUNS = {
    'implicit': (None, None, None),
    'explicit': (np.zeros(n, dtype=int), np.ones(o), np.zeros(n)),
}

STORAGE = ['coordinate', 'sparse_by_rows', 'sparse_by_columns',
           'dense_by_rows', 'dense_by_columns', 'dense']

@pytest.fixture(params=RUNS.keys())
def run(request):
    return RUNS[request.param]

@pytest.fixture(params=STORAGE)
def solution(request, run):
    return solve(request.param, *run)


def test_sllsb_status(solution):
    x, y, z, r, x_stat, inform = solution
    assert inform['status'] == 0


def test_sllsb_solution(solution):
    x, y, z, r, x_stat, inform = solution
    # returned values are consistent with the objective at x
    assert r == pytest.approx(A @ x - b, abs=1e-8)
    assert inform['obj'] == pytest.approx(objective(x))
    # first-order optimality, with multipliers computed from g (see
    # test_sllsb_multipliers); x_stat < 0 flags the active bounds
    g = gradient(x)
    check_optimality(x, g)
    assert np.all(x_stat[:n - 2] < 0) and np.all(x_stat[n - 2:] == 0)
    assert x == pytest.approx(x_star, abs=1e-6)
    assert r == pytest.approx(r_star, abs=1e-6)
    assert g - np.min(g) == pytest.approx(z_star, abs=1e-6)
    assert inform['obj'] == pytest.approx(f_star, abs=1e-6)


@pytest.mark.parametrize('w', [None, np.ones(o)], ids=['w=None', 'w=ones'])
def test_sllsb_shift(w):
    # a nonzero shift changes the solution, whether or not w is given
    x_s = np.r_[np.zeros(n - 1), 2.0]
    x, y, z, r, x_stat, inform = solve('coordinate', None, w, x_s)
    assert inform['status'] == 0
    assert inform['obj'] == pytest.approx(objective(x, w, x_s))
    g = gradient(x, w, x_s)
    check_optimality(x, g)
    # the shift adds -sigma x_s to g, so g_8 = 2 x_8 - 19, g_9 = 2 x_9 - 22.
    # x_8, x_9 cannot both be free, as g_8 = g_9 and x_8 + x_9 = 1 would
    # give x_8 = -1/4. So x = e_9, with y = g_9 = -20 and z_j = 9 - j for
    # j < 9 (ignoring the shift would give x_star instead)
    assert np.all(x_stat[:n - 1] < 0) and x_stat[n - 1] == 0
    assert x == pytest.approx(np.r_[np.zeros(n - 1), 1.0], abs=1e-6)
    assert g == pytest.approx(y[0] + z, abs=1e-6)
    assert y == pytest.approx([-20.0], abs=1e-6)
    assert z == pytest.approx(np.r_[9.0 - np.arange(n - 1), 0.0], abs=1e-6)


def test_sllsb_multipliers(solution):
    x, y, z, r, x_stat, inform = solution
    assert gradient(x) == pytest.approx(y[0] + z, abs=1e-6)
    assert y == pytest.approx(y_star, abs=1e-6)
    assert z == pytest.approx(z_star, abs=1e-6)
