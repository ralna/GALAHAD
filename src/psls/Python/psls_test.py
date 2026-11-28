from galahad import psls
import numpy as np
import pytest


# psls forms a positive definite preconditioner P of a symmetric matrix A,
# chosen by options['preconditioner'], and solves P x = b. apply_
# preconditioner overwrites b by x and returns it. Preconditioners:
# < 0: P = I; 0 (automatic, currently 1) and 1: P = diag( max( |A_ii|,
# min_diagonal ) ); 2: P = band( A ), modified to be positive definite;
# 4: A with Schnabel-Eskow modifications; 5: A = L D L^T with D replaced by
# |D| (Gill, Murray, Ponceleon and Saunders); 6: the incomplete Cholesky
# factorization ICFS of Lin and More'; 3 and 7 need HSL, and 8 is not yet
# available

# 1) as in test_psls.py: A = ( 1 0 0 ; 0 2 1 ; 0 1 3 ) is positive definite
# and A x = b = ( 1, 3, 4 ) for x = ( 1, 1, 1 ); the diagonal preconditioner
# gives x = ( 1, 3/2, 4/3 )
A = np.array([[1.0, 0.0, 0.0],
              [0.0, 2.0, 1.0],
              [0.0, 1.0, 3.0]])
b = np.array([1.0, 3.0, 4.0])
n = 3
x_exact = np.array([1.0, 1.0, 1.0])
x_diag = np.array([1.0, 1.5, 4.0 / 3.0])

# 2) A_i = ( 1 0 0 ; 0 -2 1 ; 0 1 3 ) is indefinite. A_i = L D L^T with
# L = I - e_3 e_2^T / 2 and D = diag( 1, -2, 7/2 ), so the GMPS
# preconditioner is P = L |D| L^T = ( 1 0 0 ; 0 2 -1 ; 0 -1 4 ), and the
# diagonal one P = diag( 1, 2, 3 )
A_i = np.array([[1.0, 0.0, 0.0],
                [0.0, -2.0, 1.0],
                [0.0, 1.0, 3.0]])
P_gmps = np.array([[1.0, 0.0, 0.0],
                   [0.0, 2.0, -1.0],
                   [0.0, -1.0, 4.0]])

STORAGE = ['coordinate', 'sparse_by_rows', 'dense']

def storage(A, A_type):
    # return A_ne, A_row, A_col, A_ptr, A_val for the lower triangle of A
    if A_type == 'dense':
        rows, cols = np.tril_indices(A.shape[0])
        return len(rows), None, None, None, A[rows, cols]
    rows, cols = np.nonzero(np.tril(A))
    if A_type == 'coordinate':
        return len(rows), rows, cols, None, A[rows, cols]
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=A.shape[0]))]
    return len(rows), None, cols, ptr, A[rows, cols]

def options(preconditioner, **kw):
    o = psls.initialize()
    o['print_level'] = 0
    o['preconditioner'] = preconditioner
    # use the LAPACK solvers rather than the defaults, which need the
    # environment variables OMP_CANCELLATION and OMP_PROC_BIND set
    o['symmetric_linear_solver'] = 'sytr'
    o['definite_linear_solver'] = 'potr'
    o.update(kw)
    return o

@pytest.fixture
def terminate():
    yield
    psls.terminate()

def form(A, preconditioner, A_type='coordinate', **kw):
    A_ne, A_row, A_col, A_ptr, A_val = storage(A, A_type)
    psls.load(n, A_type, A_ne, A_row, A_col, A_ptr,
              options(preconditioner, **kw))
    psls.form_preconditioner(A_ne, A_val)

def apply(rhs):
    rhs_c = rhs.copy()
    return psls.apply_preconditioner(n, rhs_c), rhs_c

def inverse_preconditioner():
    # P^-1, column by column
    return np.column_stack([apply(e)[0] for e in np.eye(n)])


@pytest.fixture
def applied(terminate):
    # problem 1) with the incomplete Cholesky preconditioner, which is exact
    # for this A
    form(A, 6)
    x, rhs_c = apply(b)
    return x, rhs_c, psls.information()

def test_psls_status(applied):
    x, rhs_c, inform = applied
    assert inform['status'] == 0
    assert inform['preconditioner'] == 6

def test_psls_solution(applied):
    x, rhs_c, inform = applied
    assert x == pytest.approx(x_exact, abs=1e-8)

def test_psls_in_place(applied):
    # apply_preconditioner overwrites b by x, and returns it
    x, rhs_c, inform = applied
    assert x is rhs_c


# problem 1) with the other preconditioners
PD = [(-1, b, -1), (0, x_diag, 1), (1, x_diag, 1), (2, x_exact, 2),
      (4, x_exact, 4), (5, x_exact, 5)]

@pytest.mark.parametrize('preconditioner, x_ref, used', PD,
                         ids=[str(p[0]) for p in PD])
def test_psls_positive_definite(terminate, preconditioner, x_ref, used):
    form(A, preconditioner)
    x, rhs_c = apply(b)
    inform = psls.information()
    assert inform['status'] == 0
    assert inform['preconditioner'] == used
    assert inform['perturbed'] is False
    assert x == pytest.approx(x_ref, abs=1e-8)

@pytest.mark.parametrize('semi_bandwidth, x_ref', [(0, x_diag), (1, x_exact)])
def test_psls_band(terminate, semi_bandwidth, x_ref):
    # band( A ) is diag( A ) for semi-bandwidth 0, and A for 1
    form(A, 2, semi_bandwidth=semi_bandwidth)
    x, rhs_c = apply(b)
    assert psls.information()['status'] == 0
    assert x == pytest.approx(x_ref, abs=1e-8)

@pytest.mark.parametrize('A_type', STORAGE)
def test_psls_storage(terminate, A_type):
    form(A, 6, A_type)
    x, rhs_c = apply(b)
    assert psls.information()['status'] == 0
    assert x == pytest.approx(x_exact, abs=1e-8)

def test_psls_reform(terminate):
    # a second formation, with A doubled
    A_ne, A_row, A_col, A_ptr, A_val = storage(A, 'coordinate')
    psls.load(n, 'coordinate', A_ne, A_row, A_col, A_ptr, options(5))
    psls.form_preconditioner(A_ne, A_val)
    psls.form_preconditioner(A_ne, 2.0 * A_val)
    x, rhs_c = apply(b)
    assert psls.information()['status'] == 0
    assert x == pytest.approx(x_exact / 2.0, abs=1e-8)


# problem 2): P must be positive definite although A_i is not
@pytest.mark.parametrize('preconditioner, P_ref', [(1, np.diag([1.0, 2.0,
                                                                 3.0])),
                                                   (5, P_gmps)],
                         ids=['1', '5'])
def test_psls_indefinite_exact(terminate, preconditioner, P_ref):
    form(A_i, preconditioner)
    assert psls.information()['status'] == 0
    assert inverse_preconditioner() == pytest.approx(np.linalg.inv(P_ref),
                                                     abs=1e-8)

@pytest.mark.parametrize('preconditioner', [
    2, 6,
    pytest.param(4, marks=pytest.mark.xfail(
        strict=True, reason='the Schnabel-Eskow preconditioner is not '
        'positive definite (see https://github.com/ralna/GALAHAD/issues/659)'))])
def test_psls_indefinite(terminate, preconditioner):
    form(A_i, preconditioner)
    assert psls.information()['status'] == 0
    P_inv = inverse_preconditioner()
    assert P_inv == pytest.approx(P_inv.T, abs=1e-12)
    assert np.linalg.eigvalsh(P_inv).min() > 0.0

def test_psls_band_perturbed(terminate):
    # the band preconditioner of A_i is perturbed to be positive definite
    form(A_i, 2)
    assert psls.information()['perturbed'] is True


# preconditioners that are unavailable: 3 and 7 need the HSL packages MC61
# and MI28 (status -26 and -80), and 8 is not yet implemented (-45). As a
# preconditioner cannot then be applied (see issue 659), only the status is
# checked
@pytest.mark.parametrize('preconditioner, status',
                         [(3, -26), (7, -80), (8, -45)])
def test_psls_unavailable(terminate, preconditioner, status):
    with pytest.warns(RuntimeWarning):
        form(A, preconditioner)
    assert psls.information()['status'] == status


# subset preconditioners: the C interface used the indices of sub as if they
# started from 1, so the wrong rows are used, possibly out of bounds, which
# cannot be a strict xfail
@pytest.mark.skip(reason='psls uses the wrong rows for a subset '
                  'preconditioner (see https://github.com/ralna/GALAHAD/pull/658)')
@pytest.mark.parametrize('preconditioner', [1, 5])
def test_psls_subset(terminate, preconditioner):
    # the preconditioner of the submatrix in rows and columns 1 and 2:
    # x_1 and x_2 solve P_sub ( x_1, x_2 ) = ( 3, 4 ), and x_0 = b_0
    sub = np.array([1, 2])
    A_ne, A_row, A_col, A_ptr, A_val = storage(A, 'coordinate')
    psls.load(n, 'coordinate', A_ne, A_row, A_col, A_ptr,
              options(preconditioner))
    psls.form_subset_preconditioner(A_ne, A_val, len(sub), sub)
    x, rhs_c = apply(b)
    assert psls.information()['status'] == 0
    A_sub = A[np.ix_(sub, sub)]
    P_sub = np.diag(np.diag(A_sub)) if preconditioner == 1 else A_sub
    assert x[0] == b[0]
    assert x[1:] == pytest.approx(np.linalg.solve(P_sub, b[1:]), abs=1e-8)
