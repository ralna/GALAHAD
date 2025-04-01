purpose
-------

The ``nodend`` package finds a symmetric row and column permutation, 
$P A P^T$, of a symmetric, sparse matrix $A$ with the aim of limiting 
the fill-in during subsequent Cholesky-like factorization. The package 
is actually a wrapper to the METIS_NodeND procedure from versions 
4.0, 5.1 and 5.2 of the METIS package from the Karypis Lab.

See Section 4 of $GALAHAD/doc/nodend.pdf for additional details.

method
------

Variants of node-based nested-dissection ordering are employed.

reference
---------

The basic methods are those described in

  G. Karypis.
  METIS, A software package for partitioning unstructured
  graphs, partitioning meshes, and computing
  fill-reducing orderings of sparse matrices, Version 5,
  Department of Computer Science & Engineering, University of Minnesota
  Minneapolis, MN 55455, USA (2013), see

  https://github.com/KarypisLab/METIS/blob/master/manual/manual.pdf
