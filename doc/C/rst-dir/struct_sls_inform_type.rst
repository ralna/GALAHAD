.. index:: pair: table; sls_inform_type
.. _doxid-structsls__inform__type:

sls_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_sls.h>
	
	struct sls_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structsls__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structsls__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structsls__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`more_info<doxid-structsls__inform__type_more_info>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`entries<doxid-structsls__inform__type_entries>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out_of_range<doxid-structsls__inform__type_out_of_range>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`duplicates<doxid-structsls__inform__type_duplicates>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`upper<doxid-structsls__inform__type_upper>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`missing_diagonals<doxid-structsls__inform__type_missing_diagonals>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_depth_assembly_tree<doxid-structsls__inform__type_max_depth_assembly_tree>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nodes_assembly_tree<doxid-structsls__inform__type_nodes_assembly_tree>`;
		int64_t :ref:`real_size_desirable<doxid-structsls__inform__type_real_size_desirable>`;
		int64_t :ref:`integer_size_desirable<doxid-structsls__inform__type_integer_size_desirable>`;
		int64_t :ref:`real_size_necessary<doxid-structsls__inform__type_real_size_necessary>`;
		int64_t :ref:`integer_size_necessary<doxid-structsls__inform__type_integer_size_necessary>`;
		int64_t :ref:`real_size_factors<doxid-structsls__inform__type_real_size_factors>`;
		int64_t :ref:`integer_size_factors<doxid-structsls__inform__type_integer_size_factors>`;
		int64_t :ref:`entries_in_factors<doxid-structsls__inform__type_entries_in_factors>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_task_pool_size<doxid-structsls__inform__type_max_task_pool_size>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_front_size<doxid-structsls__inform__type_max_front_size>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`compresses_real<doxid-structsls__inform__type_compresses_real>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`compresses_integer<doxid-structsls__inform__type_compresses_integer>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`two_by_two_pivots<doxid-structsls__inform__type_two_by_two_pivots>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`semi_bandwidth<doxid-structsls__inform__type_semi_bandwidth>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`delayed_pivots<doxid-structsls__inform__type_delayed_pivots>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`pivot_sign_changes<doxid-structsls__inform__type_pivot_sign_changes>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`static_pivots<doxid-structsls__inform__type_static_pivots>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`first_modified_pivot<doxid-structsls__inform__type_first_modified_pivot>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`rank<doxid-structsls__inform__type_rank>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`negative_eigenvalues<doxid-structsls__inform__type_negative_eigenvalues>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`num_zero<doxid-structsls__inform__type_num_zero>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iterative_refinements<doxid-structsls__inform__type_iterative_refinements>`;
		int64_t :ref:`flops_assembly<doxid-structsls__inform__type_flops_assembly>`;
		int64_t :ref:`flops_elimination<doxid-structsls__inform__type_flops_elimination>`;
		int64_t :ref:`flops_blas<doxid-structsls__inform__type_flops_blas>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`largest_modified_pivot<doxid-structsls__inform__type_largest_modified_pivot>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`minimum_scaling_factor<doxid-structsls__inform__type_minimum_scaling_factor>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`maximum_scaling_factor<doxid-structsls__inform__type_maximum_scaling_factor>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`condition_number_1<doxid-structsls__inform__type_condition_number_1>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`condition_number_2<doxid-structsls__inform__type_condition_number_2>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`backward_error_1<doxid-structsls__inform__type_backward_error_1>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`backward_error_2<doxid-structsls__inform__type_backward_error_2>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`forward_error<doxid-structsls__inform__type_forward_error>`;
		bool :ref:`alternative<doxid-structsls__inform__type_alternative>`;
		char :ref:`solver<doxid-structsls__inform__type_solver>`[21];
		struct :ref:`sls_time_type<doxid-structsls__time__type>` :ref:`time<doxid-structsls__inform__type_time>`;
		struct :ref:`sils_ainfo_type<doxid-structsils__ainfo__type>` :ref:`sils_ainfo<doxid-structsls__inform__type_sils_ainfo>`;
		struct :ref:`sils_finfo_type<doxid-structsils__finfo__type>` :ref:`sils_finfo<doxid-structsls__inform__type_sils_finfo>`;
		struct :ref:`sils_sinfo_type<doxid-structsils__sinfo__type>` :ref:`sils_sinfo<doxid-structsls__inform__type_sils_sinfo>`;
		struct ma57_ainfo :ref:`ma57_ainfo<doxid-structsls__inform__type_ma57_ainfo>`;
		struct ma57_finfo :ref:`ma57_finfo<doxid-structsls__inform__type_ma57_finfo>`;
		struct ma57_sinfo :ref:`ma57_sinfo<doxid-structsls__inform__type_ma57_sinfo>`;
		struct ma77_info :ref:`ma77_info<doxid-structsls__inform__type_ma77_info>`;
		struct ma86_info :ref:`ma86_info<doxid-structsls__inform__type_ma86_info>`;
		struct ma87_info :ref:`ma87_info<doxid-structsls__inform__type_ma87_info>`;
		struct ma97_info :ref:`ma97_info<doxid-structsls__inform__type_ma97_info>`;
		struct spral_ssids_inform :ref:`ssids_inform<doxid-structsls__inform__type_ssids_inform>`;
		struct spral_ssids_inform :ref:`nodend_inform<doxid-structsls__inform__type_nodend_inform>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mc61_info<doxid-structsls__inform__type_mc61_info>`[10];
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`mc61_rinfo<doxid-structsls__inform__type_mc61_rinfo>`[15];
		struct mc64_info :ref:`mc64_info<doxid-structsls__inform__type_mc64_info>`;
		struct mc68_info :ref:`mc68_info<doxid-structsls__inform__type_mc68_info>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mc77_info<doxid-structsls__inform__type_mc77_info>`[10];
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`mc77_rinfo<doxid-structsls__inform__type_mc77_rinfo>`[10];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mumps_error<doxid-structsls__inform__type_mumps_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mumps_info<doxid-structsls__inform__type_mumps_info>`[80];
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`mumps_rinfo<doxid-structsls__inform__type_mumps_rinfo>`[40];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`pardiso_error<doxid-structsls__inform__type_pardiso_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`pardiso_IPARM<doxid-structsls__inform__type_pardiso_IPARM>`[64];
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`pardiso_DPARM<doxid-structsls__inform__type_pardiso_DPARM>`[64];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mkl_pardiso_error<doxid-structsls__inform__type_mkl_pardiso_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mkl_pardiso_IPARM<doxid-structsls__inform__type_mkl_pardiso_IPARM>`[64];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`pastix_info<doxid-structsls__inform__type_pastix_info>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`wsmp_error<doxid-structsls__inform__type_wsmp_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`wsmp_iparm<doxid-structsls__inform__type_wsmp_iparm>`[64];
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`wsmp_dparm<doxid-structsls__inform__type_wsmp_dparm>`[64];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mpi_ierr<doxid-structsls__inform__type_mpi_ierr>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`lapack_error<doxid-structsls__inform__type_lapack_error>`;
	};
.. _details-structsls__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structsls__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

reported return status: 0 success -1 allocation error -2 deallocation error -3 matrix data faulty (.n < 1, .ne < 0) -20 alegedly +ve definite matrix is not -29 unavailable option -31 input order is not a permutation or is faulty in some other way -32 > control.max_integer_factor_size integer space required for factor -33 > control.max_real_factor_size real space required for factors -40 not possible to alter the diagonals -41 no access to permutation or pivot sequence used -42 no access to diagonal perturbations -43 direct-access file error -50 solver-specific error; see the solver's info parameter -101 unknown solver

.. index:: pair: variable; alloc_status
.. _doxid-structsls__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

STAT value after allocate failure.

.. index:: pair: variable; bad_alloc
.. _doxid-structsls__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

name of array which provoked an allocate failure

.. index:: pair: variable; more_info
.. _doxid-structsls__inform__type_more_info:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` more_info

further information on failure

.. index:: pair: variable; entries
.. _doxid-structsls__inform__type_entries:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` entries

number of entries

.. index:: pair: variable; out_of_range
.. _doxid-structsls__inform__type_out_of_range:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out_of_range

number of indices out-of-range

.. index:: pair: variable; duplicates
.. _doxid-structsls__inform__type_duplicates:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` duplicates

number of duplicates

.. index:: pair: variable; upper
.. _doxid-structsls__inform__type_upper:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` upper

number of entries from the strict upper triangle

.. index:: pair: variable; missing_diagonals
.. _doxid-structsls__inform__type_missing_diagonals:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` missing_diagonals

number of missing diagonal entries for an allegedly-definite matrix

.. index:: pair: variable; max_depth_assembly_tree
.. _doxid-structsls__inform__type_max_depth_assembly_tree:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_depth_assembly_tree

maximum depth of the assembly tree

.. index:: pair: variable; nodes_assembly_tree
.. _doxid-structsls__inform__type_nodes_assembly_tree:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nodes_assembly_tree

nodes in the assembly tree (= number of elimination steps)

.. index:: pair: variable; real_size_desirable
.. _doxid-structsls__inform__type_real_size_desirable:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t real_size_desirable

desirable or actual size for real array for the factors and other data

.. index:: pair: variable; integer_size_desirable
.. _doxid-structsls__inform__type_integer_size_desirable:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t integer_size_desirable

desirable or actual size for integer array for the factors and other dat

.. index:: pair: variable; real_size_necessary
.. _doxid-structsls__inform__type_real_size_necessary:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t real_size_necessary

necessary size for real array for the factors and other data

.. index:: pair: variable; integer_size_necessary
.. _doxid-structsls__inform__type_integer_size_necessary:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t integer_size_necessary

necessary size for integer array for the factors and other data

.. index:: pair: variable; real_size_factors
.. _doxid-structsls__inform__type_real_size_factors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t real_size_factors

predicted or actual number of reals to hold factors

.. index:: pair: variable; integer_size_factors
.. _doxid-structsls__inform__type_integer_size_factors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t integer_size_factors

predicted or actual number of integers to hold factors

.. index:: pair: variable; entries_in_factors
.. _doxid-structsls__inform__type_entries_in_factors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t entries_in_factors

number of entries in factors

.. index:: pair: variable; max_task_pool_size
.. _doxid-structsls__inform__type_max_task_pool_size:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_task_pool_size

maximum number of tasks in the factorization task pool

.. index:: pair: variable; max_front_size
.. _doxid-structsls__inform__type_max_front_size:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_front_size

forecast or actual size of largest front

.. index:: pair: variable; compresses_real
.. _doxid-structsls__inform__type_compresses_real:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` compresses_real

number of compresses of real data

.. index:: pair: variable; compresses_integer
.. _doxid-structsls__inform__type_compresses_integer:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` compresses_integer

number of compresses of integer data

.. index:: pair: variable; two_by_two_pivots
.. _doxid-structsls__inform__type_two_by_two_pivots:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` two_by_two_pivots

number of 2x2 pivots

.. index:: pair: variable; semi_bandwidth
.. _doxid-structsls__inform__type_semi_bandwidth:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` semi_bandwidth

semi-bandwidth of matrix following bandwidth reduction

.. index:: pair: variable; delayed_pivots
.. _doxid-structsls__inform__type_delayed_pivots:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` delayed_pivots

number of delayed pivots (total)

.. index:: pair: variable; pivot_sign_changes
.. _doxid-structsls__inform__type_pivot_sign_changes:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` pivot_sign_changes

number of pivot sign changes if no pivoting is used successfully

.. index:: pair: variable; static_pivots
.. _doxid-structsls__inform__type_static_pivots:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` static_pivots

number of static pivots chosen

.. index:: pair: variable; first_modified_pivot
.. _doxid-structsls__inform__type_first_modified_pivot:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` first_modified_pivot

first pivot modification when static pivoting

.. index:: pair: variable; rank
.. _doxid-structsls__inform__type_rank:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` rank

estimated rank of the matrix

.. index:: pair: variable; negative_eigenvalues
.. _doxid-structsls__inform__type_negative_eigenvalues:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` negative_eigenvalues

number of negative eigenvalues

.. index:: pair: variable; num_zero
.. _doxid-structsls__inform__type_num_zero:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` num_zero

number of pivots that are considered zero (and ignored)

.. index:: pair: variable; iterative_refinements
.. _doxid-structsls__inform__type_iterative_refinements:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iterative_refinements

number of iterative refinements performed

.. index:: pair: variable; flops_assembly
.. _doxid-structsls__inform__type_flops_assembly:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t flops_assembly

anticipated or actual number of floating-point operations in assembly

.. index:: pair: variable; flops_elimination
.. _doxid-structsls__inform__type_flops_elimination:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t flops_elimination

anticipated or actual number of floating-point operations in elimination

.. index:: pair: variable; flops_blas
.. _doxid-structsls__inform__type_flops_blas:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t flops_blas

additional number of floating-point operations for BLAS

.. index:: pair: variable; largest_modified_pivot
.. _doxid-structsls__inform__type_largest_modified_pivot:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` largest_modified_pivot

largest diagonal modification when static pivoting or ensuring definiten

.. index:: pair: variable; minimum_scaling_factor
.. _doxid-structsls__inform__type_minimum_scaling_factor:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` minimum_scaling_factor

minimum scaling factor

.. index:: pair: variable; maximum_scaling_factor
.. _doxid-structsls__inform__type_maximum_scaling_factor:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` maximum_scaling_factor

maximum scaling factor

.. index:: pair: variable; condition_number_1
.. _doxid-structsls__inform__type_condition_number_1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` condition_number_1

esimate of the condition number of the matrix (category 1 equations)

.. index:: pair: variable; condition_number_2
.. _doxid-structsls__inform__type_condition_number_2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` condition_number_2

estimate of the condition number of the matrix (category 2 equations)

.. index:: pair: variable; backward_error_1
.. _doxid-structsls__inform__type_backward_error_1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` backward_error_1

esimate of the backward error (category 1 equations)

.. index:: pair: variable; backward_error_2
.. _doxid-structsls__inform__type_backward_error_2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` backward_error_2

esimate of the backward error (category 2 equations)

.. index:: pair: variable; forward_error
.. _doxid-structsls__inform__type_forward_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` forward_error

estimate of forward error

.. index:: pair: variable; alternative
.. _doxid-structsls__inform__type_alternative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool alternative

has an "alternative" y: A y = 0 and yT b > 0 been found when trying to solve A x = b ?

.. index:: pair: variable; solver
.. _doxid-structsls__inform__type_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char solver[21]

name of external solver used to factorize and solve

.. index:: pair: variable; time
.. _doxid-structsls__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_time_type<doxid-structsls__time__type>` time

timings (see above)

.. index:: pair: variable; sils_ainfo
.. _doxid-structsls__inform__type_sils_ainfo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sils_ainfo_type<doxid-structsils__ainfo__type>` sils_ainfo

the output structure from sils

.. index:: pair: variable; sils_finfo
.. _doxid-structsls__inform__type_sils_finfo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sils_finfo_type<doxid-structsils__finfo__type>` sils_finfo

see sils_ainfo

.. index:: pair: variable; sils_sinfo
.. _doxid-structsls__inform__type_sils_sinfo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sils_sinfo_type<doxid-structsils__sinfo__type>` sils_sinfo

see sils_ainfo

.. index:: pair: variable; ma57_ainfo
.. _doxid-structsls__inform__type_ma57_ainfo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct ma57_ainfo ma57_ainfo

the output structure from ma57

.. index:: pair: variable; ma57_finfo
.. _doxid-structsls__inform__type_ma57_finfo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct ma57_finfo ma57_finfo

see ma57_ainfo

.. index:: pair: variable; ma57_sinfo
.. _doxid-structsls__inform__type_ma57_sinfo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct ma57_sinfo ma57_sinfo

see ma57_ainfo

.. index:: pair: variable; ma77_info
.. _doxid-structsls__inform__type_ma77_info:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct ma77_info ma77_info

the output structure from ma77

.. index:: pair: variable; ma86_info
.. _doxid-structsls__inform__type_ma86_info:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct ma86_info ma86_info

the output structure from ma86

.. index:: pair: variable; ma87_info
.. _doxid-structsls__inform__type_ma87_info:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct ma87_info ma87_info

the output structure from ma87

.. index:: pair: variable; ma97_info
.. _doxid-structsls__inform__type_ma97_info:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct ma97_info ma97_info

the output structure from ma97

.. index:: pair: variable; ssids_inform
.. _doxid-structsls__inform__type_ssids_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct spral_ssids_inform ssids_inform

the output structure from ssids

.. index:: pair: variable; nodend_inform
.. _doxid-structsls__inform__type_nodend_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`nodend_inform_type<doxid-structnodend__inform__type>` nodend_inform

the output structure from nodend

.. index:: pair: variable; mc61_info
.. _doxid-structsls__inform__type_mc61_info:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mc61_info[10]

the integer and real output arrays from mc61

.. index:: pair: variable; mc61_rinfo
.. _doxid-structsls__inform__type_mc61_rinfo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` mc61_rinfo[15]

see mc61_info

.. index:: pair: variable; mc64_info
.. _doxid-structsls__inform__type_mc64_info:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct mc64_info mc64_info

the output structure from mc64

.. index:: pair: variable; mc68_info
.. _doxid-structsls__inform__type_mc68_info:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct mc68_info mc68_info

the output structure from mc68

.. index:: pair: variable; mc77_info
.. _doxid-structsls__inform__type_mc77_info:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mc77_info[10]

the integer output array from mc77

.. index:: pair: variable; mc77_rinfo
.. _doxid-structsls__inform__type_mc77_rinfo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` mc77_rinfo[10]

the real output status from mc77

.. index:: pair: variable; mumps_error
.. _doxid-structsls__inform__type_mumps_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mumps_error

the output scalars and arrays from mumps

.. index:: pair: variable; mumps_info
.. _doxid-structsls__inform__type_mumps_info:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mumps_info[80]

see mumps_error

.. index:: pair: variable; mumps_rinfo
.. _doxid-structsls__inform__type_mumps_rinfo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` mumps_rinfo[40]

see mumps_error

.. index:: pair: variable; pardiso_error
.. _doxid-structsls__inform__type_pardiso_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` pardiso_error

the output scalars and arrays from pardiso

.. index:: pair: variable; pardiso_IPARM
.. _doxid-structsls__inform__type_pardiso_IPARM:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` pardiso_IPARM[64]

see pardiso_error

.. index:: pair: variable; pardiso_DPARM
.. _doxid-structsls__inform__type_pardiso_DPARM:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` pardiso_DPARM[64]

see pardiso_error

.. index:: pair: variable; mkl_pardiso_error
.. _doxid-structsls__inform__type_mkl_pardiso_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mkl_pardiso_error

the output scalars and arrays from mkl_pardiso

.. index:: pair: variable; mkl_pardiso_IPARM
.. _doxid-structsls__inform__type_mkl_pardiso_IPARM:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mkl_pardiso_IPARM[64]

see mkl_pardiso_error

.. index:: pair: variable; pastix_info
.. _doxid-structsls__inform__type_pastix_info:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` pastix_info

the output flag from pastix

.. index:: pair: variable; wsmp_error
.. _doxid-structsls__inform__type_wsmp_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` wsmp_error

the output scalars and arrays from wsmp

.. index:: pair: variable; wsmp_iparm
.. _doxid-structsls__inform__type_wsmp_iparm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` wsmp_iparm[64]

see wsmp_error

.. index:: pair: variable; wsmp_dparm
.. _doxid-structsls__inform__type_wsmp_dparm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` wsmp_dparm[64]

see wsmp_error

.. index:: pair: variable; mpi_ierr
.. _doxid-structsls__inform__type_mpi_ierr:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mpi_ierr

the output flag from MPI routines

.. index:: pair: variable; lapack_error
.. _doxid-structsls__inform__type_lapack_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` lapack_error

the output flag from LAPACK routines

