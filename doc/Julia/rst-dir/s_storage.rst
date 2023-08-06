.. _details-s_storage:

.. _details-s_storage__sym:

matrix storage
--------------

The **symmetric** $n$ by $n$ matrix $A$ may
be presented and stored in a variety of formats. But crucially symmetry
is exploited by only storing values from the *lower triangular* part
(i.e, those entries that lie on or below the leading diagonal).

*Dense* storage format:
The matrix $A$ is stored as a compact  dense matrix by rows, that
is, the values of the entries of each row in turn are stored in order
within an appropriate real one-dimensional array. Since $A$ is
symmetric, only the lower triangular part (that is the part
$A_{ij}$ for $1 \leq j \leq i \leq n$) need be held.
In this case the lower triangle should be stored by rows, that is
component $(i-1) * i / 2 + j$  of the storage array A_val
will hold the value $A_{ij}$ (and, by symmetry, $A_{ji}$)
for $1 \leq j \leq i \leq n$.
The string A_type = 'dense' should be specified.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $1 \leq l \leq ne$, of $A$,
its row index i, column index j and value $A_{ij}$,
$1 \leq j \leq i \leq n$,  are stored as the $l$-th
components of the integer arrays A_row and A_col and real array A_val,
respectively, while the number of nonzeros is recorded as A_ne = $ne$.
Note that only the entries in the lower triangle should be stored.
The string A_type = 'coordinate' should be specified.

*Sparse row-wise* storage format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $A$ the i-th component of the
integer array A_ptr holds the position of the first entry in this row,
while A_ptr(n+1) holds the total number of entries plus one.
The column indices j, $1 \leq j \leq i$, and values $A_{ij}$ of the entries
in the i-th row are stored in components l = A_ptr(i), ..., A_ptr(i+1)-1 of the
integer array A_col, and real array A_val, respectively. Note that as before
only the entries in the lower triangle should be stored. For sparse matrices,
this scheme almost always requires less storage than its predecessor.
The string A_type = 'sparse_by_rows' should be specified.

*Diagonal* storage format:
If $A$ is diagonal (i.e., $A_{ij} = 0$ for all
$1 \leq i \neq j \leq n$) only the diagonals entries
$A_{ii}$, $1 \leq i \leq n$ need be stored,
and the first n components of the array A_val may be used for the purpose.
The string A_type = 'diagonal' should be specified.

*Multiples of the identity* storage format:
If $A$ is a multiple of the identity matrix, (i.e., $H = \alpha I$
where $I$ is the n by n identity matrix and $\alpha$ is a scalar),
it suffices to store $\alpha$ as the first component of A_val.
The string A_type = 'scaled_identity' should be specified.

The *identity matrix* format:
If $A$ is the identity matrix, no values need be stored.
The string A_type = 'identity' should be specified.

The *zero matrix* format:
The same is true if $A$ is the zero matrix, but now
the string A_type = 'zero' or 'none' should be specified.
