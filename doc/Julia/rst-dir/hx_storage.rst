.. _details-hx_storage:

.. _details-hx_storage__sym:

matrix storage
--------------

The **symmetric** $n$ by $n$ matrix $H = \nabla^2_{xx}f$ may
be presented and stored in a variety of formats. But crucially symmetry
is exploited by only storing values from the lower triangular part
(i.e, those entries that lie on or below the leading diagonal).

*Dense* storage format:
The matrix $H$ is stored as a compact  dense matrix by rows, that
is, the values of the entries of each row in turn are stored in order
within an appropriate real one-dimensional array. Since $H$ is
symmetric, only the lower triangular part (that is the part
$H_{ij}$ for $1 \leq j \leq i \leq n$) need be held.
In this case the lower triangle should be stored by rows, that is
component $(i-1) * i / 2 + j$  of the storage array H_val
will hold the value $H_{ij}$ (and, by symmetry, $H_{ji}$)
for $1 \leq j \leq i \leq n$.
The string H_type = 'dense' should be specified.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $1 \leq l \leq ne$, of $H$,
its row index i, column index j and value $H_{ij}$,
$1 \leq j \leq i \leq n$,  are stored as the $l$-th
components of the integer arrays H_row and H_col and real array H_val,
respectively, while the number of nonzeros is recorded as H_ne = $ne$.
Note that only the entries in the lower triangle should be stored.
The string H_type = 'coordinate' should be specified.

*Sparse row-wise* storage format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $H$ the i-th component of the
integer array H_ptr holds the position of the first entry in this row,
while H_ptr(n+1) holds the total number of entries plus one.
The column indices j, $1 \leq j \leq i$, and values $H_{ij}$ of the entries
in the i-th row are stored in components l = H_ptr(i), ..., H_ptr(i+1)-1 of the
integer array H_col, and real array H_val, respectively. Note that as before
only the entries in the lower triangle should be stored. For sparse matrices,
this scheme almost always requires less storage than its predecessor.
The string H_type = 'sparse_by_rows' should be specified.

*Diagonal* storage format:
If $H$ is diagonal (i.e., $H_{ij} = 0$ for all
$1 \leq i \neq j \leq n$) only the diagonals entries
$H_{ii}$, $1 \leq i \leq n$ need be stored,
and the first n components of the array H_val may be used for the purpose.
The string H_type = 'diagonal' should be specified.
