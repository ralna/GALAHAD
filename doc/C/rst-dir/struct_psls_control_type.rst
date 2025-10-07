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
	
		bool :ref:`f_indexing<doxid-structpsls__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structpsls__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structpsls__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structpsls__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`preconditioner<doxid-structpsls__control__type_preconditioner>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`semi_bandwidth<doxid-structpsls__control__type_semi_bandwidth>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`scaling<doxid-structpsls__control__type_scaling>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ordering<doxid-structpsls__control__type_ordering>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_col<doxid-structpsls__control__type_max_col>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`icfs_vectors<doxid-structpsls__control__type_icfs_vectors>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mi28_lsize<doxid-structpsls__control__type_mi28_lsize>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mi28_rsize<doxid-structpsls__control__type_mi28_rsize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`min_diagonal<doxid-structpsls__control__type_min_diagonal>`;
		bool :ref:`new_structure<doxid-structpsls__control__type_new_structure>`;
		bool :ref:`get_semi_bandwidth<doxid-structpsls__control__type_get_semi_bandwidth>`;
		bool :ref:`get_norm_residual<doxid-structpsls__control__type_get_norm_residual>`;
		bool :ref:`space_critical<doxid-structpsls__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structpsls__control__type_deallocate_error_fatal>`;
		char :ref:`symmetric_linear_solver<doxid-structpsls__control__type_symmetric_linear_solver>`[31];
		char :ref:`definite_linear_solver<doxid-structpsls__control__type_definite_linear_solver>`[31];
		char :ref:`prefix<doxid-structpsls__control__type_prefix>`[31];
		struct :ref:`sls_control_type<doxid-structsls__control__type>` :ref:`sls_control<doxid-structpsls__control__type_sls_control>`;
		struct :ref:`mi28_control<doxid-structmi28__control>` :ref:`mi28_control<doxid-structpsls__control__type_mi28_control>`;
	};
.. _details-structpsls__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structpsls__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structpsls__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structpsls__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structpsls__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

controls level of diagnostic output

.. index:: pair: variable; preconditioner
.. _doxid-structpsls__control__type_preconditioner:

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
.. _doxid-structpsls__control__type_semi_bandwidth:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` semi_bandwidth

the semi-bandwidth for band(H) when .preconditioner = 2,3

.. index:: pair: variable; scaling
.. _doxid-structpsls__control__type_scaling:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` scaling

not used at present

.. index:: pair: variable; ordering
.. _doxid-structpsls__control__type_ordering:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ordering

see scaling

.. index:: pair: variable; max_col
.. _doxid-structpsls__control__type_max_col:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_col

maximum number of nonzeros in a column of $A$ for Schur-complement factorization to accommodate newly deleted rpws and columns

.. index:: pair: variable; icfs_vectors
.. _doxid-structpsls__control__type_icfs_vectors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` icfs_vectors

number of extra vectors of length n required by the Lin-More' incomplete Cholesky preconditioner when .preconditioner = 6

.. index:: pair: variable; mi28_lsize
.. _doxid-structpsls__control__type_mi28_lsize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mi28_lsize

the maximum number of fill entries within each column of the incomplete factor L computed by HSL_MI28 when .preconditioner = 7. In general, increasing mi28_lsize improve the quality of the preconditioner but increases the time to compute and then apply the preconditioner. Values less than 0 are treated as 0

.. index:: pair: variable; mi28_rsize
.. _doxid-structpsls__control__type_mi28_rsize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mi28_rsize

the maximum number of entries within each column of the strictly lower triangular matrix $R$ used in the computation of the preconditioner by HSL_MI28 when .preconditioner = 7. Rank-1 arrays of size .mi28_rsize \* n are allocated internally to hold $R$. Thus the amount of memory used, as well as the amount of work involved in computing the preconditioner, depends on mi28_rsize. Setting mi28_rsize > 0 generally leads to a higher quality preconditioner than using mi28_rsize = 0, and choosing mi28_rsize >= mi28_lsize is generally recommended

.. index:: pair: variable; min_diagonal
.. _doxid-structpsls__control__type_min_diagonal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` min_diagonal

the minimum permitted diagonal in diag(max(H,.min_diagonal))

.. index:: pair: variable; new_structure
.. _doxid-structpsls__control__type_new_structure:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool new_structure

set new_structure true if the storage structure for the input matrix has changed, and false if only the values have changed

.. index:: pair: variable; get_semi_bandwidth
.. _doxid-structpsls__control__type_get_semi_bandwidth:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool get_semi_bandwidth

set get_semi_bandwidth true if the semi-bandwidth of the submatrix is to be calculated

.. index:: pair: variable; get_norm_residual
.. _doxid-structpsls__control__type_get_norm_residual:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool get_norm_residual

set get_norm_residual true if the residual when applying the preconditioner are to be calculated

.. index:: pair: variable; space_critical
.. _doxid-structpsls__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structpsls__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structpsls__control__type_symmetric_linear_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

the name of the symmetric linear equation solver used. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma87', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl_pardiso', 'pastix', 'wsmp' and 'sytr, although only 'sytr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; definite_linear_solver
.. _doxid-structpsls__control__type_definite_linear_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char definite_linear_solver[31]

the name of the definite linear equation solver used. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl_pardiso', 'pastix', 'wsmp', 'potr',  'sytr' and 'pbtr', although only 'potr',  'sytr', 'pbtr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; prefix
.. _doxid-structpsls__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structpsls__control__type_sls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for SLS

.. index:: pair: variable; mi28_control
.. _doxid-structpsls__control__type_mi28_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`mi28_control<doxid-structmi28__control>` mi28_control

control parameters for HSL_MI28

