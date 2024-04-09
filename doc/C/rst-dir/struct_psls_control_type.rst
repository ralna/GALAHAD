.. index:: pair: table; psls_control_type
.. _doxid-structpsls__control__type:

psls_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_psls.h>
	
	struct psls_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structpsls__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structpsls__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structpsls__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structpsls__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`preconditioner<doxid-structpsls__control__type_1adf7719f1a4491459e361e80a00c55656>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`semi_bandwidth<doxid-structpsls__control__type_1abf884043df0f9c0d95bcff6fae1bf9bb>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`scaling<doxid-structpsls__control__type_1a26f0572eeeaa419eabb09dc89c00b89d>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ordering<doxid-structpsls__control__type_1a4175ebe476addcfc3433fc97c19e0708>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_col<doxid-structpsls__control__type_1abca2db33b9520095e98790d45a1be93f>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`icfs_vectors<doxid-structpsls__control__type_1adb095f545799aab1d69fcdca912d4afd>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mi28_lsize<doxid-structpsls__control__type_1a97a46af6187162b529821f79d1559827>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mi28_rsize<doxid-structpsls__control__type_1a8cd04d404e41a2a09c29eeb2de78cd85>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`min_diagonal<doxid-structpsls__control__type_1a984528c49e15a61a1d30fc8fa2d166cc>`;
		bool :ref:`new_structure<doxid-structpsls__control__type_1ada98b778d8b7622af0d49b064b56b8ba>`;
		bool :ref:`get_semi_bandwidth<doxid-structpsls__control__type_1a0e41e53b327ab70366ccb2f06a23a868>`;
		bool :ref:`get_norm_residual<doxid-structpsls__control__type_1acdcd8a05219b5c554c279137cb409a03>`;
		bool :ref:`space_critical<doxid-structpsls__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structpsls__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		char :ref:`definite_linear_solver<doxid-structpsls__control__type_1a9b46b7a8e0af020499e645bef711f634>`[31];
		char :ref:`prefix<doxid-structpsls__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
		struct :ref:`sls_control_type<doxid-structsls__control__type>` :ref:`sls_control<doxid-structpsls__control__type_1a31b308b91955ee385daacc3de00f161b>`;
		struct :ref:`mi28_control<doxid-structmi28__control>` :ref:`mi28_control<doxid-structpsls__control__type_1a0a72ba6769963a38f2428b875b1d295e>`;
	};
.. _details-structpsls__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structpsls__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structpsls__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structpsls__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structpsls__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

controls level of diagnostic output

.. index:: pair: variable; preconditioner
.. _doxid-structpsls__control__type_1adf7719f1a4491459e361e80a00c55656:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` preconditioner

which preconditioner to use:

* <0 no preconditioning occurs, $P = I$

* 0 the preconditioner is chosen automatically (forthcoming, and currently defaults to 1).

* 1 $A$ is replaced by the diagonal, $P$ = diag( max($A$, .min_diagonal ) ).

* 2 $A$ is replaced by the band $P$ = band($A$) with semi-bandwidth .semi_bandwidth.

* 3 $A$ is replaced by the reordered band $P$ = band( order($A$) ) with semi-bandwidth .semi_bandwidth, where order is chosen by the HSL package MC61 to move entries closer to the diagonal.

* 4 $P$ is a full factorization of $A$ using Schnabel-Eskow modifications, in which small or negative diagonals are made sensibly positive during the factorization.

* 5 $P$ is a full factorization of $A$ due to Gill, Murray, Ponceleon and Saunders, in which an indefinite factorization is altered to give a positive definite one.

* 6 $P$ is an incomplete Cholesky factorization of $A$ using the package ICFS due to Lin and More'.

* 7 $P$ is an incomplete factorization of $A$ implemented as HSL_MI28 from HSL.

* 8 $P$ is an incomplete factorization of $A$ due to Munskgaard (forthcoming).

* >8 treated as 0.

**N.B.** Options 3-8 may require additional external software that is not part of the package, and that must be obtained separately.

.. index:: pair: variable; semi_bandwidth
.. _doxid-structpsls__control__type_1abf884043df0f9c0d95bcff6fae1bf9bb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` semi_bandwidth

the semi-bandwidth for band(H) when .preconditioner = 2,3

.. index:: pair: variable; scaling
.. _doxid-structpsls__control__type_1a26f0572eeeaa419eabb09dc89c00b89d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` scaling

not used at present

.. index:: pair: variable; ordering
.. _doxid-structpsls__control__type_1a4175ebe476addcfc3433fc97c19e0708:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ordering

see scaling

.. index:: pair: variable; max_col
.. _doxid-structpsls__control__type_1abca2db33b9520095e98790d45a1be93f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_col

maximum number of nonzeros in a column of $A$ for Schur-complement factorization to accommodate newly deleted rpws and columns

.. index:: pair: variable; icfs_vectors
.. _doxid-structpsls__control__type_1adb095f545799aab1d69fcdca912d4afd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` icfs_vectors

number of extra vectors of length n required by the Lin-More' incomplete Cholesky preconditioner when .preconditioner = 6

.. index:: pair: variable; mi28_lsize
.. _doxid-structpsls__control__type_1a97a46af6187162b529821f79d1559827:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mi28_lsize

the maximum number of fill entries within each column of the incomplete factor L computed by HSL_MI28 when .preconditioner = 7. In general, increasing mi28_lsize improve the quality of the preconditioner but increases the time to compute and then apply the preconditioner. Values less than 0 are treated as 0

.. index:: pair: variable; mi28_rsize
.. _doxid-structpsls__control__type_1a8cd04d404e41a2a09c29eeb2de78cd85:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mi28_rsize

the maximum number of entries within each column of the strictly lower triangular matrix $R$ used in the computation of the preconditioner by HSL_MI28 when .preconditioner = 7. Rank-1 arrays of size .mi28_rsize \* n are allocated internally to hold $R$. Thus the amount of memory used, as well as the amount of work involved in computing the preconditioner, depends on mi28_rsize. Setting mi28_rsize > 0 generally leads to a higher quality preconditioner than using mi28_rsize = 0, and choosing mi28_rsize >= mi28_lsize is generally recommended

.. index:: pair: variable; min_diagonal
.. _doxid-structpsls__control__type_1a984528c49e15a61a1d30fc8fa2d166cc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` min_diagonal

the minimum permitted diagonal in diag(max(H,.min_diagonal))

.. index:: pair: variable; new_structure
.. _doxid-structpsls__control__type_1ada98b778d8b7622af0d49b064b56b8ba:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool new_structure

set new_structure true if the storage structure for the input matrix has changed, and false if only the values have changed

.. index:: pair: variable; get_semi_bandwidth
.. _doxid-structpsls__control__type_1a0e41e53b327ab70366ccb2f06a23a868:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool get_semi_bandwidth

set get_semi_bandwidth true if the semi-bandwidth of the submatrix is to be calculated

.. index:: pair: variable; get_norm_residual
.. _doxid-structpsls__control__type_1acdcd8a05219b5c554c279137cb409a03:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool get_norm_residual

set get_norm_residual true if the residual when applying the preconditioner are to be calculated

.. index:: pair: variable; space_critical
.. _doxid-structpsls__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structpsls__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; definite_linear_solver
.. _doxid-structpsls__control__type_1a9b46b7a8e0af020499e645bef711f634:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char definite_linear_solver[31]

the name of the definite linear equation solver used. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma87', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl_pardiso', 'pastix', 'wsmp', 'potr',  'sytr' and 'pbtr', although only 'potr',  'sytr', 'pbtr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; prefix
.. _doxid-structpsls__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structpsls__control__type_1a31b308b91955ee385daacc3de00f161b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for SLS

.. index:: pair: variable; mi28_control
.. _doxid-structpsls__control__type_1a0a72ba6769963a38f2428b875b1d295e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`mi28_control<doxid-structmi28__control>` mi28_control

control parameters for HSL_MI28

