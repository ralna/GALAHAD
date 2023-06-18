matrix storage
--------------

The **unsymmetric** $m$ by $n$ model matrix $A$ may be presented
and stored in a variety of convenient input formats. 

*Dense* storage format:
The matrix $A$ is stored as a compact dense matrix by rows, that is,
the values of the entries of each row in turn are
stored in order within an appropriate real one-dimensional array.
In this case, component $n \ast i + j$  of the storage array A_val
will hold the value $A_{ij}$ for $0 \leq i \leq m-1$, $0 \leq j \leq n-1$.
The string A_type = 'dense' should be specified.

*Dense by columns* storage format:
The matrix $A$ is stored as a compact dense matrix by columns, that is,
the values of the entries of each column in turn are
stored in order within an appropriate real one-dimensional array.
In this case, component $m \ast j + i$  of the storage array A_val
will hold the value $A_{ij}$ for $0 \leq i \leq m-1$, $0 \leq j \leq n-1
The string A_type = 'dense_by_columns' should be specified.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $0 \leq l \leq ne-1$, of $A$,
its row index i, column index j and value $A_{ij}$,
$0 \leq i \leq m-1$,  $0 \leq j \leq n-1$,  are stored as the $l$-th 
components of the integer arrays A_row and A_col and real array A_val, 
respectively, while the number of nonzeros is recorded as A_ne = $ne$.
The string A_type = 'coordinate'should be specified.

*Sparse row-wise storage* format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $A$ the i-th component of the
integer array A_ptr holds the position of the first entry in this row,
while A_ptr(m) holds the total number of entries.
The column indices j, $0 \leq j \leq n-1$, and values
$A_{ij}$ of the  nonzero entries in the i-th row are stored in components
l = A_ptr(i), $\ldots$, A_ptr(i+1)-1,  $0 \leq i \leq m-1$,
of the integer array A_col, and real array A_val, respectively.
For sparse matrices, this scheme almost always requires less storage than
its predecessor.
The string A_type = 'sparse_by_rows' should be specified.

*Sparse column-wise* storage format:
Once again only the nonzero entries are stored, but this time
they are ordered so that those in column j appear directly before those
in column j+1. For the j-th column of $A$ the j-th component of the
integer array A_ptr holds the position of the first entry in this column,
while A_ptr(n) holds the total number of entries.
The row indices i, $0 \leq i \leq m-1$, and values $A_{ij}$
of the  nonzero entries in the j-th columnsare stored in components
l = A_ptr(j), $\ldots$, A_ptr(j+1)-1, $0 \leq j \leq n-1$,
of the integer array A_row, and real array A_val, respectively.
As before, for sparse matrices, this scheme almost always requires less
storage than the co-ordinate format.
The string A_type = 'sparse_by_columns' should be specified.

The **symmetric** $n$ by $n$ scaing matrix $S$ may also
be presented and stored in a variety of formats. But crucially symmetry
is exploited by only storing values from the *lower triangular* part
(i.e, those entries that lie on or below the leading diagonal).

*Dense* storage format:
The matrix $S$ is stored as a compact  dense matrix by rows, that
is, the values of the entries of each row in turn are stored in order
within an appropriate real one-dimensional array. Since $S$ is
symmetric, only the lower triangular part (that is the part
$S_{ij}$ for $0 \leq j \leq i \leq n-1$) need be held.
In this case the lower triangle should be stored by rows, that is
component $i * i / 2 + j$  of the storage array S_val
will hold the value $S_{ij}$ (and, by symmetry, $S_{ji}$)
for $0 \leq j \leq i \leq n-1$.
The string S_type = 'dense' should be specified.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $0 \leq l \leq ne-1$, of $S$,
its row index i, column index j and value $S_{ij}$,
$0 \leq j \leq i \leq n-1$,  are stored as the $l$-th
components of the integer arrays S_row and S_col and real array S_val,
respectively, while the number of nonzeros is recorded as
S_ne = $ne$. Note that only the entries in the lower triangle
should be stored.
The string S_type = 'coordinate' should be specified.

*Sparse row-wise* storage format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $S$ the i-th component of the
integer array S_ptr holds the position of the first entry in this row,
while S_ptr(n) holds the total number of entries.
The column indices j, $0 \leq j \leq i$, and values
$S_{ij}$ of the  entries in the i-th row are stored in components
l = S_ptr(i), ..., S_ptr(i+1)-1 of the
integer array S_col, and real array S_val, respectively. Note that as before
only the entries in the lower triangle should be stored. For sparse matrices, 
this scheme almost always requires less storage than its predecessor.
The string S_type = 'sparse_by_rows' should be specified.

*Diagonal* storage format:
If $S$ is diagonal (i.e., $S_{ij} = 0$ for all
$0 \leq i \neq j \leq n-1$) only the diagonals entries
$S_{ii}$, $0 \leq i \leq n-1$ need be stored, 
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
