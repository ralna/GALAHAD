.. _details-fdc_storage:

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
