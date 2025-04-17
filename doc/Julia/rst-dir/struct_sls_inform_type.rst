.. index:: pair: table; sls_inform_type
.. _doxid-structsls__inform__type:

sls_inform_type structure
-------------------------

.. toctree::
	:hidden:

	struct_hsl_types.rst
	struct_ssids_types.rst

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct sls_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          more_info::INT
          entries::INT
          out_of_range::INT
          duplicates::INT
          upper::INT
          missing_diagonals::INT
          max_depth_assembly_tree::INT
          nodes_assembly_tree::INT
          real_size_desirable::Int64
          integer_size_desirable::Int64
          real_size_necessary::Int64
          integer_size_necessary::Int64
          real_size_factors::Int64
          integer_size_factors::Int64
          entries_in_factors::Int64
          max_task_pool_size::INT
          max_front_size::INT
          compresses_real::INT
          compresses_integer::INT
          two_by_two_pivots::INT
          semi_bandwidth::INT
          delayed_pivots::INT
          pivot_sign_changes::INT
          static_pivots::INT
          first_modified_pivot::INT
          rank::INT
          negative_eigenvalues::INT
          num_zero::INT
          iterative_refinements::INT
          flops_assembly::Int64
          flops_elimination::Int64
          flops_blas::Int64
          largest_modified_pivot::T
          minimum_scaling_factor::T
          maximum_scaling_factor::T
          condition_number_1::T
          condition_number_2::T
          backward_error_1::T
          backward_error_2::T
          forward_error::T
          alternative::Bool
          solver::NTuple{21,Cchar}
          time::sls_time_type{T}
          sils_ainfo::sils_ainfo_type{T,INT}
          sils_finfo::sils_finfo_type{T,INT}
          sils_sinfo::sils_sinfo_type{T,INT}
          ma57_ainfo::ma57_ainfo{T,INT}
          ma57_finfo::ma57_finfo{T,INT}
          ma57_sinfo::ma57_sinfo{T,INT}
          ma77_info::ma77_info{T,INT}
          ma86_info::ma86_info{T,INT}
          ma87_info::ma87_info{T,INT}
          ma97_info::ma97_info{T,INT}
          ssids_inform::spral_ssids_inform{INT}
          nodend_inform::nodend_inform_type{INT}
          mc61_info::NTuple{10,INT}
          mc61_rinfo::NTuple{15,T}
          mc64_info::mc64_info{INT}
          mc68_info::mc68_info{INT}
          mc77_info::NTuple{10,INT}
          mc77_rinfo::NTuple{10,T}
          mumps_error::INT
          mumps_info::NTuple{80,INT}
          mumps_rinfo::NTuple{40,T}
          pardiso_error::INT
          pardiso_IPARM::NTuple{64,INT}
          pardiso_DPARM::NTuple{64,T}
          mkl_pardiso_error::INT
          mkl_pardiso_IPARM::NTuple{64,INT}
          pastix_info::INT
          wsmp_error::INT
          wsmp_iparm::NTuple{64,INT}
          wsmp_dparm::NTuple{64,T}
          mpi_ierr::INT
          lapack_error::INT

.. _details-structsls__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structsls__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

reported return status. Possible values are:

* **0**

  success 

* **-1**

  allocation error 

* **-2** 

  deallocation error 

* **-3**

  matrix data faulty (n < 1, ne < 0) 

* **-20**

  alegedly postive definite definite matrix is indefinite

* **-29**

  unavailable option 

* **-31**

  input order is not a permutation or is faulty in some other way 

* **-32**

  > control.max_integer_factor_size integer space required for factor 

* **-33**

  > control.max_real_factor_size real space required for factors 

* **-40**

  not possible to alter the diagonals 

* **-41**

  no access to permutation or pivot sequence used 

* **-42**

  no access to diagonal perturbations 

* **-43**

  direct-access file error 

* **-50**

  solver-specific error; see the solver's info parameter 

* **-101**

  unknown solver

.. index:: pair: variable; alloc_status
.. _doxid-structsls__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

STAT value after allocate failure.

.. index:: pair: variable; bad_alloc
.. _doxid-structsls__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

name of array which provoked an allocate failure

.. index:: pair: variable; more_info
.. _doxid-structsls__inform__type_more_info:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT more_info

further information on failure

.. index:: pair: variable; entries
.. _doxid-structsls__inform__type_entries:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT entries

number of entries

.. index:: pair: variable; out_of_range
.. _doxid-structsls__inform__type_out_of_range:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out_of_range

number of indices out-of-range

.. index:: pair: variable; duplicates
.. _doxid-structsls__inform__type_duplicates:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT duplicates

number of duplicates

.. index:: pair: variable; upper
.. _doxid-structsls__inform__type_upper:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT upper

number of entries from the strict upper triangle

.. index:: pair: variable; missing_diagonals
.. _doxid-structsls__inform__type_missing_diagonals:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT missing_diagonals

number of missing diagonal entries for an allegedly-definite matrix

.. index:: pair: variable; max_depth_assembly_tree
.. _doxid-structsls__inform__type_max_depth_assembly_tree:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_depth_assembly_tree

maximum depth of the assembly tree

.. index:: pair: variable; nodes_assembly_tree
.. _doxid-structsls__inform__type_nodes_assembly_tree:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nodes_assembly_tree

nodes in the assembly tree (= number of elimination steps)

.. index:: pair: variable; real_size_desirable
.. _doxid-structsls__inform__type_real_size_desirable:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 real_size_desirable

desirable or actual size for real array for the factors and other data

.. index:: pair: variable; integer_size_desirable
.. _doxid-structsls__inform__type_integer_size_desirable:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 integer_size_desirable

desirable or actual size for integer array for the factors and other dat

.. index:: pair: variable; real_size_necessary
.. _doxid-structsls__inform__type_real_size_necessary:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 real_size_necessary

necessary size for real array for the factors and other data

.. index:: pair: variable; integer_size_necessary
.. _doxid-structsls__inform__type_integer_size_necessary:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 integer_size_necessary

necessary size for integer array for the factors and other data

.. index:: pair: variable; real_size_factors
.. _doxid-structsls__inform__type_real_size_factors:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 real_size_factors

predicted or actual number of reals to hold factors

.. index:: pair: variable; integer_size_factors
.. _doxid-structsls__inform__type_integer_size_factors:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 integer_size_factors

predicted or actual number of integers to hold factors

.. index:: pair: variable; entries_in_factors
.. _doxid-structsls__inform__type_entries_in_factors:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 entries_in_factors

number of entries in factors

.. index:: pair: variable; max_task_pool_size
.. _doxid-structsls__inform__type_max_task_pool_size:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_task_pool_size

maximum number of tasks in the factorization task pool

.. index:: pair: variable; max_front_size
.. _doxid-structsls__inform__type_max_front_size:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_front_size

forecast or actual size of largest front

.. index:: pair: variable; compresses_real
.. _doxid-structsls__inform__type_compresses_real:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT compresses_real

number of compresses of real data

.. index:: pair: variable; compresses_integer
.. _doxid-structsls__inform__type_compresses_integer:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT compresses_integer

number of compresses of integer data

.. index:: pair: variable; two_by_two_pivots
.. _doxid-structsls__inform__type_two_by_two_pivots:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT two_by_two_pivots

number of 2x2 pivots

.. index:: pair: variable; semi_bandwidth
.. _doxid-structsls__inform__type_semi_bandwidth:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT semi_bandwidth

semi-bandwidth of matrix following bandwidth reduction

.. index:: pair: variable; delayed_pivots
.. _doxid-structsls__inform__type_delayed_pivots:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT delayed_pivots

number of delayed pivots (total)

.. index:: pair: variable; pivot_sign_changes
.. _doxid-structsls__inform__type_pivot_sign_changes:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT pivot_sign_changes

number of pivot sign changes if no pivoting is used successfully

.. index:: pair: variable; static_pivots
.. _doxid-structsls__inform__type_static_pivots:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT static_pivots

number of static pivots chosen

.. index:: pair: variable; first_modified_pivot
.. _doxid-structsls__inform__type_first_modified_pivot:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT first_modified_pivot

first pivot modification when static pivoting

.. index:: pair: variable; rank
.. _doxid-structsls__inform__type_rank:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT rank

estimated rank of the matrix

.. index:: pair: variable; negative_eigenvalues
.. _doxid-structsls__inform__type_negative_eigenvalues:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT negative_eigenvalues

number of negative eigenvalues

.. index:: pair: variable; num_zero
.. _doxid-structsls__inform__type_num_zero:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT num_zero

number of pivots that are considered zero (and ignored)

.. index:: pair: variable; iterative_refinements
.. _doxid-structsls__inform__type_iterative_refinements:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iterative_refinements

number of iterative refinements performed

.. index:: pair: variable; flops_assembly
.. _doxid-structsls__inform__type_flops_assembly:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 flops_assembly

anticipated or actual number of floating-point operations in assembly

.. index:: pair: variable; flops_elimination
.. _doxid-structsls__inform__type_flops_elimination:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 flops_elimination

anticipated or actual number of floating-point operations in elimination

.. index:: pair: variable; flops_blas
.. _doxid-structsls__inform__type_flops_blas:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 flops_blas

additional number of floating-point operations for BLAS

.. index:: pair: variable; largest_modified_pivot
.. _doxid-structsls__inform__type_largest_modified_pivot:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T largest_modified_pivot

largest diagonal modification when static pivoting or ensuring definiten

.. index:: pair: variable; minimum_scaling_factor
.. _doxid-structsls__inform__type_minimum_scaling_factor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T minimum_scaling_factor

minimum scaling factor

.. index:: pair: variable; maximum_scaling_factor
.. _doxid-structsls__inform__type_maximum_scaling_factor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T maximum_scaling_factor

maximum scaling factor

.. index:: pair: variable; condition_number_1
.. _doxid-structsls__inform__type_condition_number_1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T condition_number_1

esimate of the condition number of the matrix (category 1 equations)

.. index:: pair: variable; condition_number_2
.. _doxid-structsls__inform__type_condition_number_2:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T condition_number_2

estimate of the condition number of the matrix (category 2 equations)

.. index:: pair: variable; backward_error_1
.. _doxid-structsls__inform__type_backward_error_1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T backward_error_1

esimate of the backward error (category 1 equations)

.. index:: pair: variable; backward_error_2
.. _doxid-structsls__inform__type_backward_error_2:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T backward_error_2

esimate of the backward error (category 2 equations)

.. index:: pair: variable; forward_error
.. _doxid-structsls__inform__type_forward_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T forward_error

estimate of forward error

.. index:: pair: variable; alternative
.. _doxid-structsls__inform__type_alternative:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool alternative

has an "alternative" $y$: $A y = 0$ and $y^T b > 0$ been found when trying 
to solve $A x = b$ ?

.. index:: pair: variable; solver
.. _doxid-structsls__inform__type_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char solver[21]

name of external solver used to factorize and solve

.. index:: pair: variable; time
.. _doxid-structsls__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_time_type<doxid-structsls__time__type>` time

timings (see above)

.. index:: pair: variable; sils_ainfo
.. _doxid-structsls__inform__type_sils_ainfo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sils_ainfo_type<doxid-structsils__ainfo__type>` sils_ainfo

the analyse output structure from sils

.. index:: pair: variable; sils_finfo
.. _doxid-structsls__inform__type_sils_finfo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sils_finfo_type<doxid-structsils__finfo__type>` sils_finfo

the factorize output structure from sils

.. index:: pair: variable; sils_sinfo
.. _doxid-structsls__inform__type_sils_sinfo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sils_sinfo_type<doxid-structsils__sinfo__type>` sils_sinfo

the solve output structure from sils

.. index:: pair: variable; ma57_ainfo
.. _doxid-structsls__inform__type_ma57_ainfo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma57_ainfo <details-structma57__ainfo>` ma57_ainfo

the analyse output structure from hsl_ma57

.. index:: pair: variable; ma57_finfo
.. _doxid-structsls__inform__type_ma57_finfo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma57_finfo <details-structma57__finfo>` ma57_finfo

the factorize output structure from hsl_ma57

.. index:: pair: variable; ma57_sinfo
.. _doxid-structsls__inform__type_ma57_sinfo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma57_sinfo <details-structma57__sinfo>` ma57_sinfo

the solve output structure from hsl_ma57

.. index:: pair: variable; ma77_info
.. _doxid-structsls__inform__type_ma77_info:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma77_info <details-structma77__info>` ma77_info

the output structure from hsl_ma77

.. index:: pair: variable; ma86_info
.. _doxid-structsls__inform__type_ma86_info:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma86_info <details-structma86__info>` ma86_info

the output structure from hsl_ma86

.. index:: pair: variable; ma87_info
.. _doxid-structsls__inform__type_ma87_info:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma87_info <details-structma87__info>` ma87_info

the output structure from hsl_ma87

.. index:: pair: variable; ma97_info
.. _doxid-structsls__inform__type_ma97_info:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma97_info <details-structma97__info>` ma97_info

the output structure from hsl_ma97

.. index:: pair: variable; ssids_inform
.. _doxid-structsls__inform__type_ssids_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`spral_ssids_inform <details-structspral__ssids__inform>` ssids_inform

the output structure from ssids

.. index:: pair: variable; nodend_inform
.. _doxid-structsls__inform__type_nodend_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`nodend_inform_type<doxid-structnodend__inform__type>` nodend_inform

the output structure from nodend

.. index:: pair: variable; mc61_info
.. _doxid-structsls__inform__type_mc61_info:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mc61_info[10]

the real output array from mc61 from HSL

.. index:: pair: variable; mc61_rinfo
.. _doxid-structsls__inform__type_mc61_rinfo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T mc61_rinfo[15]

the integer output array from mc61 from HSL

.. index:: pair: variable; mc64_info
.. _doxid-structsls__inform__type_mc64_info:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct  :ref:`mc64_info <details-structmc64__info>` mc64_info

the output structure from hsl_mc64

.. index:: pair: variable; mc68_info
.. _doxid-structsls__inform__type_mc68_info:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`mc68_info <details-structmc68__info>` mc68_info

the output structure from hsl_mc68

.. index:: pair: variable; mc77_info
.. _doxid-structsls__inform__type_mc77_info:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mc77_info[10]

the integer output array from mc77

.. index:: pair: variable; mc77_rinfo
.. _doxid-structsls__inform__type_mc77_rinfo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T mc77_rinfo[10]

the real output array from mc77 from HSL

.. index:: pair: variable; mumps_error
.. _doxid-structsls__inform__type_mumps_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mumps_error

the output scalars and arrays from mumps

.. index:: pair: variable; mumps_info
.. _doxid-structsls__inform__type_mumps_info:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mumps_info[80]

see mumps_error

.. index:: pair: variable; mumps_rinfo
.. _doxid-structsls__inform__type_mumps_rinfo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T mumps_rinfo[40]

see mumps_error

.. index:: pair: variable; pardiso_error
.. _doxid-structsls__inform__type_pardiso_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT pardiso_error

the output scalars and arrays from pardiso

.. index:: pair: variable; pardiso_IPARM
.. _doxid-structsls__inform__type_pardiso_IPARM:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT pardiso_IPARM[64]

see pardiso_error

.. index:: pair: variable; pardiso_DPARM
.. _doxid-structsls__inform__type_pardiso_DPARM:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pardiso_DPARM[64]

see pardiso_error

.. index:: pair: variable; mkl_pardiso_error
.. _doxid-structsls__inform__type_mkl_pardiso_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mkl_pardiso_error

the output scalars and arrays from mkl_pardiso

.. index:: pair: variable; mkl_pardiso_IPARM
.. _doxid-structsls__inform__type_mkl_pardiso_IPARM:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mkl_pardiso_IPARM[64]

see mkl_pardiso_error

.. index:: pair: variable; pastix_info
.. _doxid-structsls__inform__type_pastix_info:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT pastix_info

the output flag from pastix

.. index:: pair: variable; wsmp_error
.. _doxid-structsls__inform__type_wsmp_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT wsmp_error

the output scalars and arrays from wsmp

.. index:: pair: variable; wsmp_iparm
.. _doxid-structsls__inform__type_wsmp_iparm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT wsmp_iparm[64]

see wsmp_error

.. index:: pair: variable; wsmp_dparm
.. _doxid-structsls__inform__type_wsmp_dparm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T wsmp_dparm[64]

see wsmp_error

.. index:: pair: variable; mpi_ierr
.. _doxid-structsls__inform__type_mpi_ierr:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mpi_ierr

the output flag from MPI routines

.. index:: pair: variable; lapack_error
.. _doxid-structsls__inform__type_lapack_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT lapack_error

the output flag from LAPACK routines

