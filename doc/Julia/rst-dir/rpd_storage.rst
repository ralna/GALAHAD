.. _details-rpd_storage:

matrix storage
--------------

The $n$ by $n$ objective Hessian matrix $H$, the $n$ by $n$ 
constraint Hessians $(H_c)_i$, $i = 1, \ldots, m$ and
the $m$ by $n$ constraint Jacobian $A$ will be available in 
a sparse co-ordinate storage format.

Only the nonzero entries of the matrices are stored. For the $l$-th entry of
$A$, its row index $i$, column index $j$ and value $a_{ij}$ are stored in the 
$l$-th components of the integer arrays A_row, A_col and real array A_val, 
respectively. The order is unimportant, but the total number of entries 
A_ne is also required.

The same scheme is applicable to
$H$ (thus requiring integer arrays H_row, H_col, a real array
H_val and an integer value H_ne),
except that only the entries in the *lower triangle* need be stored.

For the constraint Hessians, a third index giving the constraint involved
is required for each entry, and is stored in the integer array
H_ptr. Once again, only the lower traingle is stored.
