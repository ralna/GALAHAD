.. _doxid-index_nodend:

GALAHAD NODEND package
======================

.. module:: galahad.nodend

.. include:: ../../Python/nodend_intro.rst

.. _details-a_storage__sym:

matrix storage
--------------

The sparsity pattern of the **symmetric** $n$ by $n$ matrix $A$ may
be presented and stored in a variety of formats. But crucially symmetry
is exploited by only storing values from the *lower triangular* part
(i.e, those entries that lie on or below the leading diagonal).

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $1 \leq l \leq ne$, of $A$,
its row index i and column index j,
$1 \leq j \leq i \leq n$,  are stored as the $l$-th
components of the integer arrays A_row and A_col,
respectively, while the number of nonzeros is recorded as A_ne = $ne$.
Note that only the entries in the lower triangle should be stored.
The string A_type = 'coordinate' should be specified.

*Sparse row-wise* storage format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $A$ the i-th component of the
integer array A_ptr holds the position of the first entry in this row,
while A_ptr(n+1) holds the total number of entries plus one.
The column indices j, $1 \leq j \leq i$, of the entries
in the i-th row are stored in components l = A_ptr(i), ..., A_ptr(i+1)-1 of the
integer array A_col. Note that as before
only the entries in the lower triangle should be stored. For sparse matrices,
this scheme almost always requires less storage than its predecessor.
The string A_type = 'sparse_by_rows' should be specified.

.. toctree::
	:hidden:

	nodend_functions.rst

.. include:: it.rst

.. include:: nodend_functions.rst

available structures
--------------------

.. include :: struct_nodend_control_type.rst

.. include :: struct_nodend_inform_type.rst

|	:ref:`genindex`

.. _doxid-index_nodend_examples:

example calls
-------------

This is an example of how to use the package to pick row/column orderings
of a given matrix $A$ prior to Cholesky-like factorization; 
the code is available in $GALAHAD/src/nodend/Julia/test_nodend.jl .
A variety of supported  matrix storage formats are shown.

.. include :: ../../../src/nodend/Julia/test_nodend.jl
   :code: julia
