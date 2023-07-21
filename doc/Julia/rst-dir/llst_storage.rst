matrix storage
--------------

*Dense* storage format:
The matrix $S$ is stored as a compact  dense matrix by rows, that
is, the values of the entries of each row in turn are stored in order
within an appropriate real one-dimensional array. Since $S$ is
symmetric, only the lower triangular part (that is the part
$S_{ij}$ for $1 \leq j \leq i \leq n$) need be held.
In this case the lower triangle should be stored by rows, that is
component $(i-1) * i / 2 + j$  of the storage array S_val
will hold the value $S_{ij}$ (and, by symmetry, $S_{ji}$)
for $1 \leq j \leq i \leq n$.
The string S_type = 'dense' should be specified.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $1 \leq l \leq ne$, of $S$,
its row index i, column index j and value $S_{ij}$,
$1 \leq j \leq i \leq n$,  are stored as the $l$-th
components of the integer arrays S_row and S_col and real array S_val,
respectively, while the number of nonzeros is recorded as S_ne = $ne$.
Note that only the entries in the lower triangle should be stored.
The string S_type = 'coordinate' should be specified.

*Sparse row-wise* storage format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $S$ the i-th component of the
integer array S_ptr holds the position of the first entry in this row,
while S_ptr(n+1) holds the total number of entries plus one.
The column indices j, $1 \leq j \leq i$, and values $S_{ij}$ of the entries
in the i-th row are stored in components l = S_ptr(i), ..., S_ptr(i+1)-1 of the
integer array S_col, and real array S_val, respectively. Note that as before
only the entries in the lower triangle should be stored. For sparse matrices,
this scheme almost always requires less storage than its predecessor.
The string S_type = 'sparse_by_rows' should be specified.

*Diagonal* storage format:
If $S$ is diagonal (i.e., $S_{ij} = 0$ for all
$1 \leq i \neq j \leq n$) only the diagonals entries
$S_{ii}$, $1 \leq i \leq n$ need be stored,
and the first n components of the array S_val may be used for the purpose.
The string S_type = 'diagonal' should be specified.

*Multiples of the identity* storage format:
If $S$ is a multiple of the identity matrix, (i.e., $H = \alpha I$
where $I$ is the n by n identity matrix and $\alpha$ is a scalar),
it suffices to store $\alpha$ as the first component of S_val.
The string S_type = 'scaled_identity' should be specified.

The *identity matrix* format:
If $S$ is the identity matrix, no values need be stored.
The string S_type = 'identity' should be specified. Strictly
this is not required as $S$ will be assumed to be $I$ if it
is not explicitly provided.

The *zero matrix* format:
The same is true if $S$ is the zero matrix, but now
the string S_type = 'zero' or 'none' should be specified.
