.. index:: pair: struct; eqp_control_type
.. _doxid-structeqp__control__type:

eqp_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_eqp.h>
	
	struct eqp_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structeqp__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structeqp__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structeqp__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structeqp__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization<doxid-structeqp__control__type_1a108359f1209601e6c6074c215e3abd8b>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_col<doxid-structeqp__control__type_1abca2db33b9520095e98790d45a1be93f>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`indmin<doxid-structeqp__control__type_1a5031bbc31f94e4cba6a540a3182b6d80>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`valmin<doxid-structeqp__control__type_1a0e142fa8dc9c363c3c2993b6129b0955>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`len_ulsmin<doxid-structeqp__control__type_1a600c95211b782597cd1b2475bb2c54c6>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`itref_max<doxid-structeqp__control__type_1a903ba4ef0869186a65d4c32459a6a0ed>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_maxit<doxid-structeqp__control__type_1a7a1029142a22f3e2a1963c3428276849>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`preconditioner<doxid-structeqp__control__type_1adf7719f1a4491459e361e80a00c55656>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`semi_bandwidth<doxid-structeqp__control__type_1abf884043df0f9c0d95bcff6fae1bf9bb>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`new_a<doxid-structeqp__control__type_1a7bea45d51fd9384037bbbf82f7750ce6>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`new_h<doxid-structeqp__control__type_1ae60c5b5b987dd62f25253ba4164813f5>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`sif_file_device<doxid-structeqp__control__type_1a65c6f8382f1e75cd0b8abd5d148188d0>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`pivot_tol<doxid-structeqp__control__type_1a133347eb5f45a24a77b63b4afd4212e8>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`pivot_tol_for_basis<doxid-structeqp__control__type_1a1912d9ec51c4e88125762b7d03ef31a6>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`zero_pivot<doxid-structeqp__control__type_1aed8525bc028ed7ae0a9dd1bb3154cda2>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`inner_fraction_opt<doxid-structeqp__control__type_1a5840187311d612d0efdfecf7078a2b7e>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`radius<doxid-structeqp__control__type_1a72757b6410f755f008e2fb6d711b61be>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`min_diagonal<doxid-structeqp__control__type_1a984528c49e15a61a1d30fc8fa2d166cc>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`max_infeasibility_relative<doxid-structeqp__control__type_1a975d28c18ad695cd215c57948fac11c0>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`max_infeasibility_absolute<doxid-structeqp__control__type_1aebbb79536b216e3c116e5a9d21426840>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`inner_stop_relative<doxid-structeqp__control__type_1a7aedce06c5903f675a7c1496f42ff834>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`inner_stop_absolute<doxid-structeqp__control__type_1a80ae9dce0bca96a4691b0a222e3257b5>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`inner_stop_inter<doxid-structeqp__control__type_1a5f1bee15068a0476f5dd4e65dc4468ce>`;
		bool :ref:`find_basis_by_transpose<doxid-structeqp__control__type_1aa88001d7f86370d329247cf28f8ff499>`;
		bool :ref:`remove_dependencies<doxid-structeqp__control__type_1ae17a6b550239434c639239ddf45bc1ad>`;
		bool :ref:`space_critical<doxid-structeqp__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structeqp__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		bool :ref:`generate_sif_file<doxid-structeqp__control__type_1aa75b3a16d146c0d7ad57bf9817033843>`;
		char :ref:`sif_file_name<doxid-structeqp__control__type_1aaa95e830b709da79d9790471bab54193>`[31];
		char :ref:`prefix<doxid-structeqp__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
		struct :ref:`fdc_control_type<doxid-structfdc__control__type>` :ref:`fdc_control<doxid-structeqp__control__type_1a7bef6e4f678e16a4dcdc40677efddd80>`;
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>` :ref:`sbls_control<doxid-structeqp__control__type_1a04ba974b3c8d21137deb070d0e8dfc3a>`;
		struct :ref:`gltr_control_type<doxid-structgltr__control__type>` :ref:`gltr_control<doxid-structeqp__control__type_1aa48d482633f3788830b1d8dc85fa91d6>`;
	};
.. _details-structeqp__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structeqp__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structeqp__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structeqp__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structeqp__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level

.. index:: pair: variable; factorization
.. _doxid-structeqp__control__type_1a108359f1209601e6c6074c215e3abd8b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization

the factorization to be used. Possible values are /li 0 automatic /li 1 Schur-complement factorization /li 2 augmented-system factorization (OBSOLETE)

.. index:: pair: variable; max_col
.. _doxid-structeqp__control__type_1abca2db33b9520095e98790d45a1be93f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_col

the maximum number of nonzeros in a column of A which is permitted with the Schur-complement factorization (OBSOLETE)

.. index:: pair: variable; indmin
.. _doxid-structeqp__control__type_1a5031bbc31f94e4cba6a540a3182b6d80:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` indmin

an initial guess as to the integer workspace required by SBLS (OBSOLETE)

.. index:: pair: variable; valmin
.. _doxid-structeqp__control__type_1a0e142fa8dc9c363c3c2993b6129b0955:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` valmin

an initial guess as to the real workspace required by SBLS (OBSOLETE)

.. index:: pair: variable; len_ulsmin
.. _doxid-structeqp__control__type_1a600c95211b782597cd1b2475bb2c54c6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` len_ulsmin

an initial guess as to the workspace required by ULS (OBSOLETE)

.. index:: pair: variable; itref_max
.. _doxid-structeqp__control__type_1a903ba4ef0869186a65d4c32459a6a0ed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` itref_max

the maximum number of iterative refinements allowed (OBSOLETE)

.. index:: pair: variable; cg_maxit
.. _doxid-structeqp__control__type_1a7a1029142a22f3e2a1963c3428276849:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_maxit

the maximum number of CG iterations allowed. If cg_maxit < 0, this number will be reset to the dimension of the system + 1

.. index:: pair: variable; preconditioner
.. _doxid-structeqp__control__type_1adf7719f1a4491459e361e80a00c55656:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` preconditioner

the preconditioner to be used for the CG. Possible values are

* 0 automatic

* 1 no preconditioner, i.e, the identity within full factorization

* 2 full factorization

* 3 band within full factorization

* 4 diagonal using the barrier terms within full factorization (OBSOLETE)

* 5 optionally supplied diagonal, G = D

.. index:: pair: variable; semi_bandwidth
.. _doxid-structeqp__control__type_1abf884043df0f9c0d95bcff6fae1bf9bb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` semi_bandwidth

the semi-bandwidth of a band preconditioner, if appropriate (OBSOLETE)

.. index:: pair: variable; new_a
.. _doxid-structeqp__control__type_1a7bea45d51fd9384037bbbf82f7750ce6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` new_a

how much has A changed since last problem solved: 0 = not changed, 1 = values changed, 2 = structure changed

.. index:: pair: variable; new_h
.. _doxid-structeqp__control__type_1ae60c5b5b987dd62f25253ba4164813f5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` new_h

how much has H changed since last problem solved: 0 = not changed, 1 = values changed, 2 = structure changed

.. index:: pair: variable; sif_file_device
.. _doxid-structeqp__control__type_1a65c6f8382f1e75cd0b8abd5d148188d0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` sif_file_device

specifies the unit number to write generated SIF file describing the current problem

.. index:: pair: variable; pivot_tol
.. _doxid-structeqp__control__type_1a133347eb5f45a24a77b63b4afd4212e8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` pivot_tol

the threshold pivot used by the matrix factorization. See the documentation for SBLS for details (OBSOLETE)

.. index:: pair: variable; pivot_tol_for_basis
.. _doxid-structeqp__control__type_1a1912d9ec51c4e88125762b7d03ef31a6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` pivot_tol_for_basis

the threshold pivot used by the matrix factorization when finding the ba See the documentation for ULS for details (OBSOLETE)

.. index:: pair: variable; zero_pivot
.. _doxid-structeqp__control__type_1aed8525bc028ed7ae0a9dd1bb3154cda2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` zero_pivot

any pivots smaller than zero_pivot in absolute value will be regarded to zero when attempting to detect linearly dependent constraints (OBSOLETE)

.. index:: pair: variable; inner_fraction_opt
.. _doxid-structeqp__control__type_1a5840187311d612d0efdfecf7078a2b7e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` inner_fraction_opt

the computed solution which gives at least inner_fraction_opt times the optimal value will be found (OBSOLETE)

.. index:: pair: variable; radius
.. _doxid-structeqp__control__type_1a72757b6410f755f008e2fb6d711b61be:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` radius

an upper bound on the permitted step (-ve will be reset to an appropriat large value by eqp_solve)

.. index:: pair: variable; min_diagonal
.. _doxid-structeqp__control__type_1a984528c49e15a61a1d30fc8fa2d166cc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` min_diagonal

diagonal preconditioners will have diagonals no smaller than min_diagonal (OBSOLETE)

.. index:: pair: variable; max_infeasibility_relative
.. _doxid-structeqp__control__type_1a975d28c18ad695cd215c57948fac11c0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` max_infeasibility_relative

if the constraints are believed to be rank defficient and the residual at a "typical" feasible point is larger than max( max_infeasibility_relative \* norm A, max_infeasibility_absolute ) the problem will be marked as infeasible

.. index:: pair: variable; max_infeasibility_absolute
.. _doxid-structeqp__control__type_1aebbb79536b216e3c116e5a9d21426840:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` max_infeasibility_absolute

see max_infeasibility_relative

.. index:: pair: variable; inner_stop_relative
.. _doxid-structeqp__control__type_1a7aedce06c5903f675a7c1496f42ff834:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` inner_stop_relative

the computed solution is considered as an acceptable approximation to th minimizer of the problem if the gradient of the objective in the preconditioning(inverse) norm is less than max( inner_stop_relative \* initial preconditioning(inverse) gradient norm, inner_stop_absolute )

.. index:: pair: variable; inner_stop_absolute
.. _doxid-structeqp__control__type_1a80ae9dce0bca96a4691b0a222e3257b5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` inner_stop_absolute

see inner_stop_relative

.. index:: pair: variable; inner_stop_inter
.. _doxid-structeqp__control__type_1a5f1bee15068a0476f5dd4e65dc4468ce:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` inner_stop_inter

see inner_stop_relative

.. index:: pair: variable; find_basis_by_transpose
.. _doxid-structeqp__control__type_1aa88001d7f86370d329247cf28f8ff499:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool find_basis_by_transpose

if .find_basis_by_transpose is true, implicit factorization precondition will be based on a basis of A found by examining A's transpose (OBSOLETE)

.. index:: pair: variable; remove_dependencies
.. _doxid-structeqp__control__type_1ae17a6b550239434c639239ddf45bc1ad:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool remove_dependencies

if .remove_dependencies is true, the equality constraints will be preprocessed to remove any linear dependencies

.. index:: pair: variable; space_critical
.. _doxid-structeqp__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structeqp__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structeqp__control__type_1aa75b3a16d146c0d7ad57bf9817033843:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_sif_file

if .generate_sif_file is .true. if a SIF file describing the current problem is to be generated

.. index:: pair: variable; sif_file_name
.. _doxid-structeqp__control__type_1aaa95e830b709da79d9790471bab54193:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char sif_file_name[31]

name of generated SIF file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structeqp__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; fdc_control
.. _doxid-structeqp__control__type_1a7bef6e4f678e16a4dcdc40677efddd80:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fdc_control_type<doxid-structfdc__control__type>` fdc_control

control parameters for FDC

.. index:: pair: variable; sbls_control
.. _doxid-structeqp__control__type_1a04ba974b3c8d21137deb070d0e8dfc3a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for SBLS

.. index:: pair: variable; gltr_control
.. _doxid-structeqp__control__type_1aa48d482633f3788830b1d8dc85fa91d6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`gltr_control_type<doxid-structgltr__control__type>` gltr_control

control parameters for GLTR

