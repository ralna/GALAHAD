from galahad import rpd
import numpy as np
import pytest


# rpd reads the data for a quadratic program from a file in QPLIB format.
# The problem below is the ALLINIT example from $GALAHAD/examples, with
# non-default initial values for x, y and z: minimize
#   1/2 x^T H x + g^T x + f
# subject to c_l <= A x + 1/2 x^T H_c x <= c_u and x_l <= x <= x_u, with
# H tridiagonal ( 2 on the diagonal, -1 off it ), g = ( -0.2, -0.4, ...,
# -1.0 ), f = 0, a quadratic term 2 x_1^2 in the first constraint,
# A = ( 1 0 1 0 0 ; 0 1 0 1 0 ), c_l = 1, c_u = infinity, x_l = 0, x_u = 2,
# x_4 integer and x_5 binary
QPLIB = """! example problem
ALLINIT   problem name
QGQ       # problem is a mixed quadratic program with a quadratic constraint
Minimize  # minimize the objective function
5         # variables
2         # general linear constraints

9         # nonzeros in lower trangle H
1 1 2.0
2 1 -1.0
2 2 2.0
3 2 -1.0
3 3 2.0
4 3 -1.0
4 4 2.0
5 4 -1.0
5 5 2.0

-0.2      default value for entries in g
4         # non default entries in g
2 -0.4
3 -0.6
4 -0.8
5 -1.0

0.0       value of f

1         # nonzeros in H_c
1 1 1 2.0

4         # nonzeros in A
1 1 1.0
1 3 1.0
2 2 1.0
2 4 1.0

1.0E+20   infinity

1.0       default value for entries in c_l
0         # non default entries in c_l

1.0E+20   default value for entries in c_u
0         # non default entries in c_u

0.0       default value for entries in x_l
0         # non default entries in x_l

2.0       default value for entries in x_u
0         # non default entries in x_u

0         default variable type
2         # non default types
4  1
5  2

0.0       default value for initial values for x
2         # non default entries in x
1 1.5
3 0.5

0.0       default value for initial values for y
1         # non default entries in y
2 -1.0

1.0       default value for initial values for z
1         # non default entries in z
5 0.0

0         # non default names for variables

0         # non default names for constraints
"""
n, m = 5, 2
INFINITY = 1.0e20
RPD_PR = 'https://github.com/ralna/GALAHAD/pull/660'

@pytest.fixture
def qplib_file(tmp_path):
    path = tmp_path / 'ALLINIT.qplib'
    path.write_text(QPLIB)
    return str(path)

@pytest.fixture
def loaded(qplib_file):
    options = rpd.initialize()
    options['print_level'] = 0
    stats = rpd.get_stats(qplib_file, options)
    yield stats
    rpd.terminate()


def test_rpd_stats(loaded):
    assert loaded[1:] == (n, m, 9, 4, 1)
    assert rpd.information()['status'] == 0

@pytest.mark.xfail(strict=True, reason='rpd_get_stats returns the first '
                   f'character of the problem type three times (see {RPD_PR})')
def test_rpd_type(loaded):
    assert loaded[0] == 'QGQ'

def test_rpd_objective(loaded):
    assert rpd.get_g(n) == pytest.approx([-0.2, -0.4, -0.6, -0.8, -1.0])
    assert rpd.get_f() == 0.0
    # the lower triangle of H, with indices from 0
    H_row, H_col, H_val = rpd.get_h(9)
    H = np.zeros((n, n))
    H[H_row, H_col] = H_val
    H_ref = 2.0 * np.eye(n) - np.eye(n, k=-1)
    assert np.array_equal(H, H_ref)

def test_rpd_constraints(loaded):
    A_row, A_col, A_val = rpd.get_a(4)
    A = np.zeros((m, n))
    A[A_row, A_col] = A_val
    assert np.array_equal(A, [[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]])
    H_c_ptr, H_c_row, H_c_col, H_c_val = rpd.get_h_c(1)
    assert list(H_c_ptr) == [0]
    assert list(H_c_row) == [0]
    assert list(H_c_col) == [0]
    assert list(H_c_val) == [2.0]
    c_l, c_u = rpd.get_clu(m)
    assert list(c_l) == [1.0, 1.0]
    assert list(c_u) == [INFINITY, INFINITY]

def test_rpd_variables(loaded):
    x_l, x_u = rpd.get_xlu(n)
    assert list(x_l) == [0.0] * n
    assert list(x_u) == [2.0] * n
    assert list(rpd.get_x_type(n)) == [0, 0, 0, 1, 2]

def test_rpd_initial_values(loaded):
    assert list(rpd.get_y(m)) == [0.0, -1.0]
    assert list(rpd.get_z(n)) == [1.0, 1.0, 1.0, 1.0, 0.0]

@pytest.mark.xfail(strict=True, reason='rpd_get_x returns g (see '
                   f'{RPD_PR})')
def test_rpd_initial_x(loaded):
    assert list(rpd.get_x(n)) == [1.5, 0.0, 0.5, 0.0, 0.0]
