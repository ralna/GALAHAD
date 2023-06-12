purpose
-------

The ``bsc`` package takes given matrices $A$ and (diagonal) $D$, and
**builds the Schur complement** $S = A D A^T$ in sparse co-ordinate 
(and optionally sparse column) format(s). Full advantage is taken 
of any zero coefficients in the matrix $A$.

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.
Please contact us if you would like full functionality!

See Section 4 of $GALAHAD/doc/bsc.pdf for a brief description of the
method employed and other details.
