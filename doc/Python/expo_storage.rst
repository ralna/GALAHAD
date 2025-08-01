matrix storage
--------------

The **unsymmetric** $m$ by $n$ Jacobian matrix $J = J(x)$
may be presentedand stored in a variety of convenient input formats. 

*Dense* storage format:
The matrix $J$ is stored as a compact  dense matrix by rows, that is,
the values of the entries of each row in turn are
stored in order within an appropriate real one-dimensional array.
In this case, component $n \ast i + j$  of the storage array J_val
will hold the value $J_{ij}$ for $0 \leq i \leq m-1$, $0 \leq j \leq n-1$.

*Dense by columns* storage format:
The matrix $J$ is stored as a compact  dense matrix by columns, that is,
the values of the entries of each column in turn are
stored in order within an appropriate real one-dimensional array.
In this case, component $m \ast j + i$  of the storage array J_val
will hold the value $J_{ij}$ for $0 \leq i \leq m-1$, $0 \leq j \leq n-1$.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $0 \leq l \leq ne-1$, of $J$,
its row index i, column index j and value $J_{ij}$,
$0 \leq i \leq m-1$,  $0 \leq j \leq n-1$,  are stored as
the $l$-th components of the integer arrays J_row and
J_col and real array J_val, respectively, while the number of nonzeros
is recorded as J_ne = $ne$.

*Sparse row-wise storage* format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $J$ the i-th component of the
integer array J_ptr holds the position of the first entry in this row,
while J_ptr(m) holds the total number of entries.
The column indices j, $0 \leq j \leq n-1$, and values
$J_{ij}$ of the  nonzero entries in the i-th row are stored in components
l = J_ptr(i), $\ldots$, J_ptr(i+1)-1,  $0 \leq i \leq m-1$,
of the integer array J_col, and real array J_val, respectively.
For sparse matrices, this scheme almost always requires less storage than
its predecessor.

*Sparse column-wise* storage format:
Once again only the nonzero entries are stored, but this time
they are ordered so that those in column j appear directly before those
in column j+1. For the j-th column of $J$ the j-th component of the
integer array J_ptr holds the position of the first entry in this column,
while J_ptr(n) holds the total number of entries.
The row indices i, $0 \leq i \leq m-1$, and values $J_{ij}$
of the  nonzero entries in the j-th columnsare stored in components
l = J_ptr(j), $\ldots$, J_ptr(j+1)-1, $0 \leq j \leq n-1$,
of the integer array J_row, and real array J_val, respectively.
As before, for sparse matrices, this scheme almost always requires less
storage than the co-ordinate format.

The **symmetric** $n$ by $n$ matrix $H = Hl(x,y)$ may
be presented and stored in a variety of formats. But crucially symmetry
is exploited by only storing values from the lower triangular part
(i.e, those entries that lie on or below the leading diagonal).

*Dense* storage format:
The matrix $H$ is stored as a compact  dense matrix by rows, that
is, the values of the entries of each row in turn are stored in order
within an appropriate real one-dimensional array. Since $H$ is
symmetric, only the lower triangular part (that is the part
$H_{ij}$ for $0 \leq j \leq i \leq n-1$) need be held.
In this case the lower triangle should be stored by rows, that is
component $i * i / 2 + j$  of the storage array H_val
will hold the value $H_{ij}$ (and, by symmetry, $H_{ji}$)
for $0 \leq j \leq i \leq n-1$.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $0 \leq l \leq ne-1$, of $H$,
its row index i, column index j and value $H_{ij}$,
$0 \leq j \leq i \leq n-1$,  are stored as the $l$-th
components of the integer arrays H_row and H_col and real array H_val,
respectively, while the number of nonzeros is recorded as
H_ne = $ne$. Note that only the entries in the lower triangle
should be stored.

*Sparse row-wise* storage format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $H$ the i-th component of the
integer array H_ptr holds the position of the first entry in this row,
while H_ptr(n) holds the total number of entries.
The column indices j, $0 \leq j \leq i$, and values
$H_{ij}$ of the  entries in the i-th row are stored in components
l = H_ptr(i), ..., H_ptr(i+1)-1 of the
integer array H_col, and real array H_val, respectively. Note that
as before only the entries in the lower triangle should be stored. For
sparse matrices, this scheme almost always requires less storage than
its predecessor.

*Diagonal* storage format:
If $H$ is diagonal (i.e., $H_{ij} = 0$ for all
$0 \leq i \neq j \leq n-1$) only the diagonals entries
$H_{ii}$, $0 \leq i \leq n-1$ need
be stored, and the first n components of the array H_val may be
used for the purpose.

*Multiples of the identity* storage format:
If $H$ is a multiple of the identity matrix, (i.e., $H = \alpha I$
where $I$ is the n by n identity matrix and $\alpha$ is a scalar),
it suffices to store $\alpha$ as the first component of H_val.

The *identity matrix* format:
If $H$ is the identity matrix, no values need be stored.

The *zero matrix* format:
The same is true if $H$ is the zero matrix.
