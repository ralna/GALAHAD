purpose
-------

The ``bsc`` package takes given matrices $A$ and (diagonal) $D$, and
**builds the Schur complement** $S = A D A^T$ in sparse co-ordinate 
(and optionally sparse column) format(s). Full advantage is taken 
of any zero coefficients in the matrix $A$.

See Section 4 of $GALAHAD/doc/bsc.pdf for a brief description of the
method employed and other details.
