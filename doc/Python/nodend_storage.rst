matrix storage
--------------

The sparsity pattern of the **symmetric** $n$ by $n$ sparse matrix $A$ may
be presented and stored in a couple of formats. But crucially symmetry
is exploited by only indices from the *lower triangular* part
(i.e, those entries that lie on or below the leading diagonal).

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $0 \leq l \leq ne-1$, of $H$,
its row index i and column index j,
$0 \leq j \leq i \leq n-1$,  are stored as the $l$-th
components of the integer arrays H_row and H_col,
respectively, while the number of nonzeros is recorded as
H_ne = $ne$. Note that only the entries in the lower triangle
should be stored.
The string H_type = 'coordinate' should be specified.

*Sparse row-wise* storage format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $H$ the i-th component of the
integer array H_ptr holds the position of the first entry in this row,
while H_ptr(n) holds the total number of entries.
The column indices j, $0 \leq j \leq i$, are stored in components
l = H_ptr(i), ..., H_ptr(i+1)-1 of the integer array H_col. Note that as before
only the entries in the lower triangle should be stored. For sparse matrices, 
this scheme almost always requires less storage than its predecessor.
The string H_type = 'sparse_by_rows' should be specified.

