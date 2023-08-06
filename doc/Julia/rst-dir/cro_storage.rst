.. _details-cro_storage:

matrix storage
--------------

The **unsymmetric** $m$ by $n$ matrix $A$ must be presented
and stored in *sparse row-wise storage* format.
For this, only the nonzero entries are stored, and they are
ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $A$ the i-th component of the
integer array A_ptr holds the position of the first entry in this row,
while A_ptr(m+1) holds the total number of entries plus one.
The column indices j, $1 \leq j \leq n$, and values
$A_{ij}$ of the  nonzero entries in the i-th row are stored in components
l = A_ptr(i), $\ldots$, A_ptr(i+1)-1,  $1 \leq i \leq m$,
of the integer array A_col, and real array A_val, respectively.

The **symmetric** $n$ by $n$ matrix $H$ must also be presented
and stored in *sparse row-wise storage* format. But, crucially, now symmetry
is exploited by only storing values from the *lower triangular* part
(i.e, those entries that lie on or below the leading diagonal).
As before, only the nonzero entries of the matrices are stored.
Only the nonzero entries from the lower triangle are stored, and
these are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $H$ the i-th component of the
integer array H_ptr holds the position of the first entry in this row,
while H_ptr(n+1) holds the total number of entries plus one.
The column indices j, $1 \leq j \leq i$, and values
$H_{ij}$ of the  entries in the i-th row are stored in components
l = H_ptr(i), ..., H_ptr(i+1)-1 of the
integer array H_col, and real array H_val, respectively.
